"""
v3 benchmark + validation runner for the exact A* hull-correction solver.

Usage:
    python3 v3_benchmark.py validate     # exactness battery vs full convolution ground truth
    python3 v3_benchmark.py sweep        # timing regression sweep N=1000..20000
    python3 v3_benchmark.py gurobi       # sweep with Gurobi comparison (needs gurobipy + license)

Timing convention: ours = max(per-block Level-1 table time) + Level-2 solver time
(parallel-L1 convention, same as all previous reports).

Provenance of tonight's speedups (all exactness-preserving; validated 12,543/12,543
vs exact convolution after every change):
  1. greedy drift correction  - residual distributed over lambda-tied increments,
                                trajectory now sums to m exactly
  2. prefix/suffix DPs        - leave-one-out (d=1) queries answered in O(W) combines
                                instead of O(B*W) full passes; suffix built lazily
  3. clip-aware exactness     - an unclipped windowed pass IS the full conv (early stop)
  4. restricted-hull tree     - tighter verification LB from hulls of restricted tables
  5. integrality rounding     - integer costs => RC integer => ceil(LB) is sound;
                                closes fractional-relaxation gaps (Chvatal-style cut).
                                Auto-detected, only applied when tables are integral.
"""
import sys, time, collections
import numpy as np
from v3_core import gen_instance, gen_wide, block_table, conv_list
from v3_solver import solve_v3


def tabs_of(inst):
    return [block_table(inst['c'][idx], A, bvec, inst['w'][idx])
            for idx, A, bvec in zip(inst['blocks'], inst['As'], inst['bs'])]


def beta_of(G, frac):
    fin = np.where(G < 1e17)[0]
    return int(fin[int(frac * (len(fin) - 1))])


def validate(n_bench=60, n_wide=120):
    fams = [('bench', lambda r: gen_instance(int(r.randint(40, 90)), int(r.randint(10**6))), n_bench, 11),
            ('wide',  lambda r: gen_wide(int(r.randint(10**6))), n_wide, 7)]
    all_ok = True
    for fam, gen, n_tr, rs in fams:
        rng = np.random.RandomState(rs)
        tot = bad = cert = 0
        depths = collections.Counter()
        t0 = time.perf_counter()
        for _ in range(n_tr):
            inst = gen(rng)
            tabs = tabs_of(inst)
            G = conv_list(tabs)
            for beta in range(len(G)):
                if G[beta] >= 1e17:
                    continue
                r = solve_v3(tabs, beta)
                tot += 1
                cert += r['certified']
                depths[r['depth']] += 1
                if abs(r['obj'] - G[beta]) > 1e-6:
                    bad += 1
                    all_ok = False
                    print(f"  FAIL seed={inst['seed']} beta={beta} got={r['obj']} truth={G[beta]}")
        print(f"{fam}: {tot} cases, mismatches={bad}, certified={cert}/{tot}, "
              f"depths={dict(sorted(depths.items()))}, {time.perf_counter()-t0:.1f}s")
    print("VALIDATION", "PASSED" if all_ok else "FAILED")
    return all_ok


def sweep(with_gurobi=False):
    if with_gurobi:
        import gurobipy as gp
        from gurobipy import GRB

        def solve_gurobi(inst, beta, raw):
            m = gp.Model()
            m.Params.OutputFlag = 0
            m.Params.TimeLimit = 15
            m.Params.MIPGap = 0
            if raw:
                m.Params.Presolve = 0
                m.Params.Cuts = 0
                m.Params.Heuristics = 0
            x = m.addVars(inst['N'], vtype=GRB.BINARY)
            m.setObjective(gp.quicksum(inst['c'][i] * x[i] for i in range(inst['N'])), GRB.MINIMIZE)
            for idx, A, bvec in zip(inst['blocks'], inst['As'], inst['bs']):
                for r_ in range(A.shape[0]):
                    m.addConstr(gp.quicksum(A[r_, j] * x[idx[j]] for j in range(len(idx))) <= bvec[r_])
            m.addConstr(gp.quicksum(int(inst['w'][i]) * x[i] for i in range(inst['N'])) == beta)
            t0 = time.perf_counter()
            m.optimize()
            return dict(time=time.perf_counter() - t0,
                        tl=m.Status == GRB.TIME_LIMIT,
                        obj=m.ObjVal if m.SolCount else None)

    cases = [(N, s, f) for N in (1000, 5000, 10000, 20000) for f in (0.5, 0.7) for s in (1, 2)]
    hdr = f"{'N':>6} {'sd':>3} {'frac':>4} | {'L1par':>7} {'L2':>7} {'total':>7} | d cert ev"
    hdr += " | grb(def) grb(raw)" if with_gurobi else ""
    hdr += " | truth"
    print(hdr)
    worst = 0.0
    for N, seed, frac in cases:
        inst = gen_instance(N, seed)
        tb, tabs = [], []
        for idx, A, bvec in zip(inst['blocks'], inst['As'], inst['bs']):
            t0 = time.perf_counter()
            tabs.append(block_table(inst['c'][idx], A, bvec, inst['w'][idx]))
            tb.append(time.perf_counter() - t0)
        G = conv_list(tabs)
        beta = beta_of(G, frac)
        t0 = time.perf_counter()
        r = solve_v3(tabs, beta)
        t2 = time.perf_counter() - t0
        ours = max(tb) + t2
        worst = max(worst, ours)
        ok = abs(r['obj'] - G[beta]) <= 1e-6
        line = (f"{N:6d} {seed:3d} {frac:4.1f} | {max(tb):7.4f} {t2:7.3f} {ours:7.3f} | "
                f"{r['depth']} {'Y' if r['certified'] else 'n':>4} {r['exact_evals']:>2}")
        if with_gurobi:
            gd = solve_gurobi(inst, beta, raw=False)
            gr = solve_gurobi(inst, beta, raw=True)
            gds = '  TLIMIT' if gd['tl'] else f"{gd['time']:8.3f}"
            grs = '  TLIMIT' if gr['tl'] else f"{gr['time']:8.3f}"
            line += f" | {gds} {grs}"
            if gd['obj'] is not None and abs(gd['obj'] - r['obj']) > 1e-6 and not gd['tl']:
                line += "  GRB-DISAGREES"
        line += f" | {'Y' if ok else 'NO'}"
        print(line)
    print(f"\nworst ours total: {worst:.3f}s")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'validate'
    if mode == 'validate':
        validate()
    elif mode == 'sweep':
        sweep(with_gurobi=False)
    elif mode == 'gurobi':
        sweep(with_gurobi=True)
    else:
        print(__doc__)
