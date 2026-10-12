"""
depth_hunter.py  —  large-scale search for instances whose exact hull-correction
                    depth exceeds 3 (the conjectured ceiling).

WHAT "DEPTH" MEANS
------------------
For block b with exact table T_b and lower convex hull H_b, an allocation s_b is
"deviating" if T_b[s_b] > H_b[s_b] (it sits strictly above the block's own hull).
For a coupling target S, the depth d(S) is the MINIMUM number of deviating blocks
over all optimal allocations achieving G[S] = min sum_b T_b[s_b] s.t. sum s_b = S.

The hull k-way merge (Theorem B) computes the cost exactly when d=0; every extra
unit of depth is one block the convex relaxation gets wrong. Empirically d<=3 over
241M cases. This script pushes that to the billions, exactly (no solver involved):
it computes G by full min-plus convolution (ground truth) and d by an exact
polynomial DP over (coupling-sum, #deviating-blocks). If any case shows d>=4 it is
saved immediately with full data for inspection.

This is a CORRECTNESS PROBE, not a benchmark: it uses exact convolution throughout,
so a "no d>=4 found" result is a genuine certificate over the sampled instances,
independent of the fast solver.

USAGE
-----
    python depth_hunter.py                 # default: 1e9 cases, wide family, checkpoints
    python depth_hunter.py --cases 2e9     # target ~2 billion (instance,S) cases
    python depth_hunter.py --family bench  # use sir's benchmark family instead of wide
    python depth_hunter.py --seed0 1000000 # start instance seed offset (resume disjoint run)
    python depth_hunter.py --kmax 6        # search depth up to 6 (default 6; >3 is the point)

Output: progress lines every ~30s, a running depth histogram, and on exit a summary
plus depth_hunt_overflow.json containing any d>=4 specimens (empty file if none).
Safe to Ctrl-C: writes a checkpoint + summary before quitting.
"""
import sys, json, time, argparse, signal
import numpy as np
from v3_core import gen_instance, gen_wide, block_table, lower_hull, conv_list

EPS = 1e-9
BIG = 1e17


def tabs_of(inst):
    return [block_table(inst['c'][idx], A, bvec, inst['w'][idx])
            for idx, A, bvec in zip(inst['blocks'], inst['As'], inst['bs'])]


def depth_profile(tabs, kmax):
    """Exact depth for EVERY achievable coupling sum S in one instance.

    Returns (G, dmin) where G[S] is the true optimum (full convolution) and
    dmin[S] is the minimum number of deviating blocks over optimal allocations,
    capped at kmax (a value of kmax means '>= kmax'). Unreachable S -> dmin=-1.

    DP state: f[j, k] = min cost using processed blocks with coupling-sum j and
    exactly k deviating blocks (k clipped at kmax). O(B * S * kmax * maxblock).
    Then dmin[S] = smallest k with f[S, k] == G[S].
    """
    G = conv_list(tabs)
    S_hi = len(G) - 1
    # f[j,k]: min cost at sum j with k deviations (k in 0..kmax)
    f = np.full((S_hi + 1, kmax + 1), np.inf)
    f[0, 0] = 0.0
    for T in tabs:
        H = lower_hull(T)
        nf = np.full((S_hi + 1, kmax + 1), np.inf)
        smax = min(len(T) - 1, S_hi)
        for s in range(smax + 1):
            if T[s] >= BIG:
                continue
            dev = 1 if (H[s] < BIG and T[s] - H[s] > EPS) else 0
            c = T[s]
            hi = S_hi - s
            if dev == 0:
                # shift in sum by s, same k
                src = f[:hi + 1, :]
                tgt = nf[s:s + hi + 1, :]
                np.minimum(tgt, src + c, out=tgt)
            else:
                # k -> min(k+1, kmax)
                for k in range(kmax + 1):
                    nk = min(k + 1, kmax)
                    src = f[:hi + 1, k]
                    tgt = nf[s:s + hi + 1, nk]
                    np.minimum(tgt, src + c, out=tgt)
        f = nf
    dmin = np.full(S_hi + 1, -1, dtype=np.int64)
    for S in range(S_hi + 1):
        if G[S] >= BIG:
            continue
        col = f[S]
        hit = np.where(col <= G[S] + EPS)[0]
        dmin[S] = int(hit[0]) if len(hit) else kmax
    return G, dmin


class Hunter:
    def __init__(self, family, kmax, seed0):
        self.family = family
        self.kmax = kmax
        self.seed0 = seed0
        self.gen = (lambda s: gen_wide(s)) if family == 'wide' else (lambda s: gen_instance(int(np.random.RandomState(s).randint(40, 90)), s))
        self.depth_hist = {k: 0 for k in range(kmax + 1)}
        self.cases = 0
        self.instances = 0
        self.overflow = []     # d >= 4 specimens
        self.max_depth_seen = 0
        self.t0 = time.perf_counter()

    def record_instance(self, seed):
        inst = self.gen(seed)
        tabs = tabs_of(inst)
        G, dmin = depth_profile(tabs, self.kmax)
        for S in range(len(dmin)):
            d = int(dmin[S])
            if d < 0:
                continue
            self.cases += 1
            self.depth_hist[d] = self.depth_hist.get(d, 0) + 1
            if d > self.max_depth_seen:
                self.max_depth_seen = d
            if d >= 4:
                self.overflow.append(dict(
                    seed=int(seed), S=int(S), G=float(G[S]), depth=d,
                    blocks=[dict(c=list(map(float, inst['c'][idx])),
                                 w=list(map(int, inst['w'][idx])),
                                 A=[list(map(float, row)) for row in A],
                                 b=list(map(float, bvec)))
                            for idx, A, bvec in zip(inst['blocks'], inst['As'], inst['bs'])]))
                print(f"\n  *** DEPTH-{d} FOUND  seed={seed} S={S} G={G[S]} — saved ***", flush=True)
        self.instances += 1

    def summary_line(self):
        el = time.perf_counter() - self.t0
        rate = self.cases / el if el > 0 else 0
        hist = {k: v for k, v in sorted(self.depth_hist.items()) if v or k <= self.max_depth_seen}
        return (f"[{el/3600:.2f}h] instances={self.instances:,} cases={self.cases:,} "
                f"({rate:,.0f}/s) depth={hist} max_d={self.max_depth_seen} "
                f"d>=4={len(self.overflow)}")

    def dump(self, path='depth_hunt_overflow.json'):
        json.dump(self.overflow, open(path, 'w'))
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=float, default=1e9, help='target (instance,S) cases')
    ap.add_argument('--family', choices=['wide', 'bench'], default='wide')
    ap.add_argument('--kmax', type=int, default=6, help='max depth to resolve (>3 is the point)')
    ap.add_argument('--seed0', type=int, default=0, help='starting seed offset')
    ap.add_argument('--report_every', type=float, default=30.0, help='seconds between progress lines')
    args = ap.parse_args()

    H = Hunter(args.family, args.kmax, args.seed0)
    target = int(args.cases)

    def on_sigint(sig, frame):
        print("\n[interrupted]")
        print(H.summary_line())
        p = H.dump()
        print(f"overflow specimens ({len(H.overflow)}) -> {p}")
        sys.exit(0)
    signal.signal(signal.SIGINT, on_sigint)

    print(f"depth hunter: family={args.family} kmax={args.kmax} target_cases={target:,}")
    print(f"(exact convolution ground truth; searching for depth >= 4)\n", flush=True)

    last_report = time.perf_counter()
    seed = args.seed0
    while H.cases < target:
        H.record_instance(seed)
        seed += 1
        now = time.perf_counter()
        if now - last_report >= args.report_every:
            print(H.summary_line(), flush=True)
            H.dump()           # checkpoint overflow continuously
            last_report = now

    print("\n" + "=" * 70)
    print("HUNT COMPLETE")
    print(H.summary_line())
    p = H.dump()
    if H.overflow:
        print(f"\n!!! {len(H.overflow)} instance(s) with depth >= 4 — see {p}")
    else:
        print(f"\nNo depth>=4 found in {H.cases:,} cases. Conjecture d<=3 holds on this sample.")


if __name__ == '__main__':
    main()
