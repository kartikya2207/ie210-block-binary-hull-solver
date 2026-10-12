"""
exp_timing_extended.py  —  overnight experiment #2: how does the solver SCALE, and does it
stay exact, as N grows and parameters widen?

Distinct from the depth experiment: here instances are LARGE (many blocks, large N), so we
measure wall-clock time of the solver. Ground truth is checked two ways:
  - while still tractable, against full convolution conv_list (exact);
  - at all sizes, we record the solver's own certificate (certified True/False).

Timing convention matches all prior reports:
    ours = max(per-block Level-1 table time) + Level-2 solve_v3 time   (parallel-L1).

For each (N target, config) it builds one instance whose total variable count is ~N (by
appending random blocks until N is reached, using the config's block-size and weight rules),
times the solver, and verifies. It sweeps N and weight ceilings so you can see whether wide
weights blow up the correction term (they increase depth, which increases f(d)).

USAGE
    python exp_timing_extended.py                          # default sweep
    python exp_timing_extended.py --Ns 1000 5000 10000 20000 40000 --wmaxs 5 20 50
    python exp_timing_extended.py --seeds 1 2 3 --truth_cap 4000

Notes:
  - conv_list ground truth is only run when N <= truth_cap (else it is skipped; the solver's
    certificate is still recorded). Default truth_cap=3000.
  - Safe to Ctrl-C. Results print incrementally and a CSV is written to exp_timing_results.csv.
"""
import sys, time, argparse, csv
import numpy as np
from v3_core import block_table, conv_list, gen_wide
from v3_solver import solve_v3

BIG = 1e17


def build_instance(N_target, seed, n_range, wmax, A_lo, A_hi, tight=(0.25, 0.80), neg_rhs_jitter=0.0):
    """Append random blocks until total vars >= N_target."""
    rng = np.random.RandomState(seed)
    blocks, cs, ws, As, bs = [], [], [], [], []
    i = 0
    while i < N_target:
        n = min(rng.randint(*n_range), N_target - i) if (N_target - i) >= n_range[0] else (N_target - i)
        n = max(1, n)
        blocks.append(list(range(i, i + n))); i += n
        cs.append(rng.randint(-10, 11, n).astype(float))
        ws.append(rng.randint(1, wmax + 1, n))
        r = rng.randint(1, 4)
        A = rng.randint(A_lo, A_hi + 1, (r, n)).astype(float)
        bvec = A.sum(1) * rng.uniform(*tight)
        if neg_rhs_jitter > 0:
            bvec = bvec + rng.uniform(-neg_rhs_jitter, neg_rhs_jitter, r)
        As.append(A); bs.append(bvec)
    return dict(N=i, c=np.concatenate(cs), w=np.concatenate(ws), blocks=blocks, As=As, bs=bs, seed=seed)


def run_case(N, seed, wmax, A_lo, A_hi, frac, truth_cap, neg_jitter):
    inst = build_instance(N, seed, (8, 14), wmax, A_lo, A_hi, neg_rhs_jitter=neg_jitter)
    # build tables, timing per-block (parallel-L1 convention = max single-block time)
    tb, tabs = [], []
    for idx, A, bvec in zip(inst['blocks'], inst['As'], inst['bs']):
        t0 = time.perf_counter()
        tabs.append(block_table(inst['c'][idx], A, bvec, inst['w'][idx]))
        tb.append(time.perf_counter() - t0)
    # choose beta from the achievable range via a cheap reachability pass on supports
    # (use solver-internal reachability by trying a fractional index into total weight)
    total_w = sum(int(np.max(np.where(T < BIG)[0])) for T in tabs)
    beta = int(frac * total_w)
    # solve + time
    t0 = time.perf_counter()
    r = solve_v3(tabs, beta)
    t2 = time.perf_counter() - t0
    L1 = max(tb); ours = L1 + t2
    # ground truth if tractable
    truth = None; match = None
    if inst['N'] <= truth_cap:
        G = conv_list(tabs)
        if beta < len(G) and G[beta] < BIG:
            truth = float(G[beta]); match = abs(r['obj'] - truth) <= 1e-6
        elif r['obj'] >= BIG:
            truth = None; match = True   # both agree it's unreachable
    return dict(N=inst['N'], B=len(tabs), seed=seed, wmax=wmax, frac=frac, beta=beta,
                L1=L1, L2=t2, ours=ours, depth=r['depth'], certified=r['certified'],
                obj=r['obj'], truth=truth, match=match)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ns', type=int, nargs='*', default=[1000, 5000, 10000, 20000, 40000])
    ap.add_argument('--wmaxs', type=int, nargs='*', default=[5, 20, 50])
    ap.add_argument('--seeds', type=int, nargs='*', default=[1, 2])
    ap.add_argument('--fracs', type=float, nargs='*', default=[0.5, 0.7])
    ap.add_argument('--A_lo', type=int, default=0)
    ap.add_argument('--A_hi', type=int, default=3)
    ap.add_argument('--neg_jitter', type=float, default=0.0)
    ap.add_argument('--truth_cap', type=int, default=3000, help='max N for conv_list exactness check')
    args = ap.parse_args()

    rows = []
    hdr = f"{'N':>6} {'B':>4} {'wmax':>4} {'sd':>3} {'frac':>4} | {'L1':>7} {'L2':>8} {'total':>8} | d cert | match"
    print("extended timing+scaling experiment")
    print(f"Ns={args.Ns} wmaxs={args.wmaxs} seeds={args.seeds} fracs={args.fracs} "
          f"A in [{args.A_lo},{args.A_hi}] jitter={args.neg_jitter} truth_cap={args.truth_cap}\n")
    print(hdr)
    worst = 0.0
    try:
        for wmax in args.wmaxs:
            for N in args.Ns:
                for seed in args.seeds:
                    for frac in args.fracs:
                        res = run_case(N, seed, wmax, args.A_lo, args.A_hi, frac,
                                       args.truth_cap, args.neg_jitter)
                        rows.append(res); worst = max(worst, res['ours'])
                        m = {True: 'Y', False: 'NO', None: '-'}[res['match']]
                        print(f"{res['N']:6d} {res['B']:4d} {wmax:4d} {seed:3d} {frac:4.1f} | "
                              f"{res['L1']:7.4f} {res['L2']:8.3f} {res['ours']:8.3f} | "
                              f"{res['depth']} {'Y' if res['certified'] else 'n':>4} | {m}", flush=True)
    except KeyboardInterrupt:
        print("\n[interrupted]")
    # write CSV
    with open('exp_timing_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ['N','B','wmax','seed','frac','beta','L1','L2','ours','depth','certified','obj','truth','match'])
        w.writeheader(); w.writerows(rows)
    nmatch = sum(1 for r in rows if r['match'] is True)
    nchecked = sum(1 for r in rows if r['match'] is not None)
    nbad = sum(1 for r in rows if r['match'] is False)
    print(f"\nworst total: {worst:.3f}s | exactness-checked: {nmatch}/{nchecked} match, {nbad} MISMATCH")
    print("results -> exp_timing_results.csv")
    if nbad:
        print("!!! mismatches at large N -> solver returned wrong objective; inspect those rows")


if __name__ == '__main__':
    main()
