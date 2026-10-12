"""
verify_all.py  —  INDEPENDENT verification battery for the general-binary pipeline.

Purpose: prove the results are real, not self-consistent illusions. Every check here
uses FRESH code (separate enumeration, raw-x brute force) rather than re-running the
same functions, and the depth detector is tested on CONSTRUCTED depth>=4 instances —
the one regime random sampling never reaches.

Run:  python verify_all.py
Expect every line to end in OK / 0 bad. Runtime ~3-4 min.

Checks
------
1. block_table       vs an independent bit-mask enumeration (different code path)
2. conv_list (truth) vs raw-x brute force over the WHOLE instance (no block tables)
3. lower_hull        is a genuine lower convex hull (<= table, slopes nondecreasing
                     w.r.t. actual spacing, endpoints achievable)
4. depth_profile     vs independent brute-force depth at EVERY reachable S (small instances)
5. solve_v3          obj == truth, and its certified depth == depth_profile's d_min
6. CONSTRUCTED depth 4 & 5 (superincreasing-weight gadgets): brute force AND depth_profile
   both report the forced depth, and match across the full S-sweep. This confirms the
   hunter can actually DETECT d>=4 — random tests never exercise this.
"""
import numpy as np, itertools, collections, sys
from v3_core import block_table, lower_hull, conv_list, gen_wide
from v3_solver import solve_v3
from depth_hunter import depth_profile, tabs_of
BIG = 1e17; EPS = 1e-9
fails = []


def block_table_indep(c, A, bvec, w):
    n = len(c); best = {}
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)])
        if np.all(A @ x <= bvec + 1e-12):
            s = int(w @ x); v = float(c @ x)
            if s not in best or v < best[s]: best[s] = v
    T = np.full(int(w.sum()) + 1, np.inf)
    for s, v in best.items(): T[s] = v
    return T


def raw_global_G(bd):
    per = []
    for c, A, bvec, w in bd:
        n = len(c); L = {}
        for mask in range(1 << n):
            x = np.array([(mask >> i) & 1 for i in range(n)])
            if np.all(A @ x <= bvec + 1e-12):
                s = int(w @ x); v = float(c @ x)
                if s not in L or v < L[s]: L[s] = v
        per.append(L)
    G = {0: 0.0}
    for L in per:
        nG = {}
        for s0, v0 in G.items():
            for s1, v1 in L.items():
                s = s0 + s1; v = v0 + v1
                if s not in nG or v < nG[s]: nG[s] = v
        G = nG
    return G


def brute_depth_profile(tabs):
    hulls = [lower_hull(T) for T in tabs]
    opts = [[(s, float(T[s]), 1 if (H[s] < BIG and T[s] - H[s] > EPS) else 0)
             for s in range(len(T)) if T[s] < BIG] for T, H in zip(tabs, hulls)]
    best = {}
    for combo in itertools.product(*opts):
        S = sum(o[0] for o in combo); cost = sum(o[1] for o in combo); dev = sum(o[2] for o in combo)
        if S not in best or cost < best[S][0] - EPS: best[S] = (cost, dev)
        elif abs(cost - best[S][0]) <= EPS: best[S] = (best[S][0], min(best[S][1], dev))
    return best


def check1():
    rng = np.random.RandomState(0); bad = 0
    for _ in range(2000):
        n = rng.randint(2, 11); c = rng.randint(-10, 11, n).astype(float); w = rng.randint(1, 5, n)
        r = rng.randint(1, 4); A = rng.randint(0, 4, (r, n)).astype(float); bvec = A.sum(1) * rng.uniform(0.2, 0.9, r)
        T1 = block_table(c, A, bvec, w); T2 = block_table_indep(c, A, bvec, w)
        m = max(len(T1), len(T2)); a = np.full(m, np.inf); b = np.full(m, np.inf); a[:len(T1)] = T1; b[:len(T2)] = T2
        if np.any((a < BIG) != (b < BIG)) or np.any(np.abs(np.where(a < BIG, a, 0) - np.where(b < BIG, b, 0)) > 1e-9):
            bad += 1
    ok = bad == 0; print(f"CHECK 1  block_table vs independent enum: 2000/{2000} {'OK' if ok else f'{bad} BAD'}")
    return ok


def check2():
    rng = np.random.RandomState(1); bad = tot = 0
    for _ in range(300):
        B = rng.randint(2, 5); bd = []
        for _ in range(B):
            n = rng.randint(2, 8); c = rng.randint(-10, 11, n).astype(float); w = rng.randint(1, 5, n)
            r = rng.randint(1, 4); A = rng.randint(0, 4, (r, n)).astype(float); bvec = A.sum(1) * rng.uniform(0.2, 0.9, r)
            bd.append((c, A, bvec, w))
        G = conv_list([block_table(*b) for b in bd]); Graw = raw_global_G(bd)
        for s, v in Graw.items():
            tot += 1
            if s >= len(G) or G[s] >= BIG or abs(G[s] - v) > 1e-9: bad += 1
    ok = bad == 0; print(f"CHECK 2  conv_list vs raw-x brute force: {tot - bad}/{tot} {'OK' if ok else f'{bad} BAD'}")
    return ok


def check3():
    rng = np.random.RandomState(2); bad = 0
    for _ in range(3000):
        n = rng.randint(2, 11); c = rng.randint(-10, 11, n).astype(float); w = rng.randint(1, 5, n)
        r = rng.randint(1, 4); A = rng.randint(0, 4, (r, n)).astype(float); bvec = A.sum(1) * rng.uniform(0.2, 0.9, r)
        T = block_table(c, A, bvec, w); H = lower_hull(T); fin = np.where(T < BIG)[0]
        if np.any(H[fin] - T[fin] > 1e-9): bad += 1; continue
        slopes = np.diff(H[fin]) / np.diff(fin)
        if np.any(np.diff(slopes) < -1e-7): bad += 1; continue
        if abs(H[fin[0]] - T[fin[0]]) > 1e-9 or abs(H[fin[-1]] - T[fin[-1]]) > 1e-9: bad += 1
    ok = bad == 0; print(f"CHECK 3  lower_hull is a true lower convex hull: {3000 - bad}/3000 {'OK' if ok else f'{bad} BAD'}")
    return ok


def check4():
    rng = np.random.RandomState(3); bad = tot = 0; hist = collections.Counter()
    for _ in range(400):
        B = rng.randint(2, 5); bd = []
        for _ in range(B):
            n = rng.randint(2, 7); c = rng.randint(-10, 11, n).astype(float); w = rng.randint(1, 4, n)
            r = rng.randint(1, 3); A = rng.randint(0, 4, (r, n)).astype(float); bvec = A.sum(1) * rng.uniform(0.2, 0.9, r)
            bd.append((c, A, bvec, w))
        tabs = [block_table(*b) for b in bd]
        psz = 1
        for T in tabs: psz *= max(1, int((T < BIG).sum()))
        if psz > 3_000_000: continue
        G, dmin = depth_profile(tabs, 6); bf = brute_depth_profile(tabs)
        for S, (cost, dev) in bf.items():
            if S >= len(dmin): continue
            tot += 1; hist[min(dev, 6)] += 1
            if abs(G[S] - cost) > 1e-6 or dmin[S] != min(dev, 6): bad += 1
    ok = bad == 0; print(f"CHECK 4  depth_profile vs brute-force depth: {tot - bad}/{tot} {'OK' if ok else f'{bad} BAD'} | depths exercised={dict(sorted(hist.items()))}")
    return ok


def check5():
    rng = np.random.RandomState(4); bo = bd = tot = 0
    for _ in range(120):
        tabs = tabs_of(gen_wide(int(rng.randint(10**6))))
        G, dmin = depth_profile(tabs, 6)
        for S in range(len(G)):
            if G[S] >= BIG: continue
            r = solve_v3(tabs, S); tot += 1
            if abs(r['obj'] - G[S]) > 1e-6: bo += 1
            if r['certified'] and r['depth'] != dmin[S]: bd += 1
    ok = bo == 0 and bd == 0
    print(f"CHECK 5  solve_v3 obj==truth & certified-depth==d_min: {tot} cases, obj_bad={bo}, depth_bad={bd} {'OK' if ok else 'BAD'}")
    return ok


def check6():
    """Constructed forced-depth instances — the detector must report d>=4 when it exists."""
    def mid_dip_block(W):
        c = np.array([1.0, 0.0]); w = np.array([W, 2 * W]); A = np.array([[1.0, 1.0]]); bvec = np.array([1.0])
        return c, A, bvec, w
    ok = True
    for Ws, want in [([1, 3, 9, 27], 4), ([1, 3, 9, 27, 81], 5)]:
        tabs = [block_table(*mid_dip_block(W)) for W in Ws]
        S = sum(Ws)
        bf = brute_depth_profile(tabs).get(S)
        G, dmin = depth_profile(tabs, 10)
        good = bf is not None and abs(G[S] - bf[0]) < 1e-6 and dmin[S] == bf[1] == want
        ok = ok and good
        print(f"CHECK 6  constructed depth-{want}: brute=(cost {bf[0]:.0f}, d {bf[1]}), "
              f"depth_profile d_min={dmin[S]} {'OK' if good else 'BAD'}")
    # full-S sweep on the d=4 case
    tabs = [block_table(*mid_dip_block(W)) for W in [1, 3, 9, 27]]
    G, dmin = depth_profile(tabs, 10); bfp = brute_depth_profile(tabs); sweep_ok = True
    for S in range(len(G)):
        if G[S] >= BIG: continue
        bf = bfp.get(S)
        if bf is None or abs(G[S] - bf[0]) > 1e-6 or dmin[S] != bf[1]: sweep_ok = False
    print(f"CHECK 6b d=4 full-S sweep vs brute force: {'ALL MATCH OK' if sweep_ok else 'MISMATCH BAD'}")
    return ok and sweep_ok


if __name__ == '__main__':
    print("INDEPENDENT VERIFICATION BATTERY\n" + "=" * 60)
    results = [check1(), check2(), check3(), check4(), check5(), check6()]
    print("=" * 60)
    if all(results):
        print("ALL CHECKS PASSED — pipeline verified by independent means, incl. depth>=4 detection.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED — DO NOT TRUST RESULTS until resolved.")
        sys.exit(1)
