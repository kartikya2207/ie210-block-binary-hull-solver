"""
exp_depth_extended.py  —  overnight experiment #1: does depth stay small, and is the
solver still exact, OUTSIDE the favorable small-weight / nonnegative-A regime?

For each named configuration it samples random instances and, per reachable coupling sum S:
  - computes the EXACT optimum G[S] by full min-plus convolution (ground truth)
  - computes the EXACT minimum depth d(S) by the polynomial depth-DP (depth_profile)
  - runs solve_v3 and records (a) whether its objective matches ground truth, and
    (b) whether it certified.

Two things are measured per config:
  1. DEPTH DISTRIBUTION  — does d>3 appear once weights spread / A goes negative?
  2. SOLVER AUDIT        — among CERTIFIED cases, is the solver ever wrong? (soundness)
                           among ALL cases, how often is it wrong / uncertified?

This is the honest "are we general enough" test. The depth-DP and conv ground truth are
exact (validated 568/568 + 803/803 + constructed d=4/5 elsewhere), so any solver
disagreement is a real solver limitation, not a measurement artifact.

USAGE
    python exp_depth_extended.py                       # all configs, 2000 inst each
    python exp_depth_extended.py --configs wide_weights negative_A --inst 20000
    python exp_depth_extended.py --inst 50000 --kmax 8 --report 30

Instances are kept small (few small blocks) so conv_list ground truth stays tractable;
that is required for an HONEST exactness check. Skips instances whose table product is too
large (logged). Safe to Ctrl-C; writes a summary first. Saves any solver-wrong specimen to
exp_depth_wrong.json and any d>=4 specimen to exp_depth_highdepth.json.
"""
import sys, json, time, argparse, signal, collections
import numpy as np
from v3_core import conv_list
from v3_solver import solve_v3
from depth_hunter import depth_profile
from gen_extended import gen_extended, tabs_of, CONFIGS

BIG = 1e17
EPS = 1e-6


def product_size(tabs):
    p = 1
    for T in tabs:
        p *= max(1, int((T < BIG).sum()))
        if p > 10**9:
            break
    return p


def serialize(inst):
    return dict(seed=int(inst['seed']),
                blocks=[dict(c=list(map(float, inst['c'][idx])),
                             w=list(map(int, inst['w'][idx])),
                             A=[list(map(float, row)) for row in A],
                             b=list(map(float, bvec)))
                        for idx, A, bvec in zip(inst['blocks'], inst['As'], inst['bs'])])


class ConfigRun:
    def __init__(self, name, kwargs, kmax):
        self.name = name; self.kwargs = kwargs; self.kmax = kmax
        self.depth = collections.Counter()
        self.cases = 0; self.instances = 0; self.skipped = 0
        self.cert_tot = 0; self.cert_wrong = 0
        self.uncert_tot = 0; self.uncert_wrong = 0
        self.maxd = 0
        self.highdepth = []   # d>=4 specimens
        self.wrong = []       # solver-wrong specimens (esp. certified-wrong = critical)

    def step(self, seed, prod_cap):
        inst = gen_extended(seed, **self.kwargs)
        try:
            tabs = tabs_of(inst)
        except Exception:
            self.skipped += 1; return
        if product_size(tabs) > prod_cap:
            self.skipped += 1; return
        try:
            G, dmin = depth_profile(tabs, self.kmax)
        except Exception:
            self.skipped += 1; return
        feas = False
        for S in range(len(dmin)):
            d = int(dmin[S])
            if d < 0:
                continue
            feas = True
            self.cases += 1; self.depth[d] += 1
            if d > self.maxd:
                self.maxd = d
            if d >= 4 and len(self.highdepth) < 200:
                self.highdepth.append(dict(S=int(S), G=float(G[S]), depth=d, **serialize(inst)))
            r = solve_v3(tabs, S)
            wrong = abs(r['obj'] - G[S]) > EPS
            if r['certified']:
                self.cert_tot += 1
                if wrong:
                    self.cert_wrong += 1
                    if len(self.wrong) < 200:
                        self.wrong.append(dict(S=int(S), true=float(G[S]), got=float(r['obj']),
                                               certified=True, depth=int(d), **serialize(inst)))
            else:
                self.uncert_tot += 1
                if wrong:
                    self.uncert_wrong += 1
        self.instances += feas

    def line(self):
        d = dict(sorted(self.depth.items()))
        sound = "SOUND" if self.cert_wrong == 0 else f"UNSOUND({self.cert_wrong})"
        return (f"  [{self.name:14s}] inst={self.instances:>6,} cases={self.cases:>9,} "
                f"maxd={self.maxd} depth={d} | certified={self.cert_tot:,}(wrong {self.cert_wrong} {sound}) "
                f"uncert={self.uncert_tot:,}(wrong {self.uncert_wrong}) skip={self.skipped:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='*', default=list(CONFIGS.keys()))
    ap.add_argument('--inst', type=int, default=2000, help='instances per config')
    ap.add_argument('--kmax', type=int, default=8)
    ap.add_argument('--prod_cap', type=float, default=4e6, help='max table-product per instance (ground-truth tractability)')
    ap.add_argument('--report', type=float, default=30.0)
    args = ap.parse_args()

    runs = {c: ConfigRun(c, CONFIGS[c], args.kmax) for c in args.configs}

    def dump():
        hd = []; wr = []
        for r in runs.values():
            hd += r.highdepth; wr += r.wrong
        json.dump(hd, open('exp_depth_highdepth.json', 'w'))
        json.dump(wr, open('exp_depth_wrong.json', 'w'))

    def on_sigint(s, f):
        print("\n[interrupted]")
        for r in runs.values(): print(r.line())
        dump(); print("specimens -> exp_depth_highdepth.json, exp_depth_wrong.json")
        sys.exit(0)
    signal.signal(signal.SIGINT, on_sigint)

    print(f"extended depth+soundness experiment | configs={args.configs} | {args.inst} inst each | kmax={args.kmax}")
    print("(exact convolution ground truth + exact depth DP; auditing solve_v3)\n")
    t0 = time.perf_counter(); last = t0
    for c in args.configs:
        r = runs[c]
        for seed in range(args.inst):
            r.step(seed, int(args.prod_cap))
            now = time.perf_counter()
            if now - last >= args.report:
                el = now - t0
                print(f"[{el/3600:.2f}h] running {c} ({seed+1}/{args.inst})")
                print(r.line()); dump(); last = now
        print(f"--- {c} done ---"); print(r.line(), flush=True)

    print("\n" + "=" * 70 + "\nEXTENDED EXPERIMENT COMPLETE")
    any_unsound = False
    for r in runs.values():
        print(r.line())
        any_unsound |= (r.cert_wrong > 0)
    dump()
    print("\nKEY QUESTIONS:")
    print(f"  depth>3 appeared in: {[c for c,r in runs.items() if r.maxd>3]}")
    print(f"  solver SOUND (certified never wrong) in all configs: {not any_unsound}")
    if any_unsound:
        print("  !!! certified-wrong cases exist -> see exp_depth_wrong.json (this is a real solver bug)")
    print("  (uncertified-wrong is expected where true depth exceeds the solver's reach)")


if __name__ == '__main__':
    main()
