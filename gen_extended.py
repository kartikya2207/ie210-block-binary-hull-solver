"""
gen_extended.py  —  parameterized instance generator for the extended experiments.

The original families (gen_instance, gen_wide in v3_core) used small weights (<=5) and
nonnegative A. The experiments below showed depth<=3 there. This generator EXPOSES the
parameters that were previously fixed, so we can test whether the favorable findings
(low depth, exactness, speed) survive when we stress them:

  - block count B        (depth <= B, so more blocks => more room for high depth)
  - block size n_b
  - weight ceiling wmax  (wide/spread weights are what the d>=4 gadgets needed)
  - A entry range        (negative entries break the downward-closed feasibility that
                          nonnegative A quietly relied on)
  - constraint tightness

Every instance is i.i.d. given the seed; nothing is fixed across instances except the rules.
"""
import numpy as np
from v3_core import block_table

BIG = 1e17


def gen_extended(seed, B_range=(3, 9), n_range=(4, 9), wmax=5,
                 A_lo=0, A_hi=3, tight=(0.25, 0.80), neg_rhs_jitter=0.0,
                 cost_lo=-10, cost_hi=10):
    """One random instance. Ranges are [lo, hi) for randint where shown.

    neg_rhs_jitter>0 adds Uniform(-j, j) to each RHS, needed for feasibility when A has
    negative entries (otherwise A.sum(1)*tight can be trivially infeasible/degenerate).
    """
    rng = np.random.RandomState(seed)
    B = rng.randint(*B_range)
    blocks, cs, ws, As, bs = [], [], [], [], []
    i = 0
    for _ in range(B):
        n = rng.randint(*n_range)
        blocks.append(list(range(i, i + n))); i += n
        cs.append(rng.randint(cost_lo, cost_hi + 1, n).astype(float))
        ws.append(rng.randint(1, wmax + 1, n))
        r = rng.randint(1, 4)
        A = rng.randint(A_lo, A_hi + 1, (r, n)).astype(float)
        bvec = A.sum(1) * rng.uniform(*tight)
        if neg_rhs_jitter > 0:
            bvec = bvec + rng.uniform(-neg_rhs_jitter, neg_rhs_jitter, r)
        As.append(A); bs.append(bvec)
    return dict(N=i, B=B, c=np.concatenate(cs), w=np.concatenate(ws),
                blocks=blocks, As=As, bs=bs, seed=seed)


def tabs_of(inst):
    return [block_table(inst['c'][idx], A, bvec, inst['w'][idx])
            for idx, A, bvec in zip(inst['blocks'], inst['As'], inst['bs'])]


# Named configurations used by the experiment scripts.
CONFIGS = {
    # baseline reproduces the wide family's regime (small weights, nonneg A)
    'baseline':      dict(B_range=(3, 9),  n_range=(4, 9), wmax=5,  A_lo=0,  A_hi=3),
    # push block count up (depth <= B)
    'many_blocks':   dict(B_range=(8, 16), n_range=(4, 9), wmax=5,  A_lo=0,  A_hi=3),
    # spread the weights (the d>=4 gadgets needed spread weights)
    'wide_weights':  dict(B_range=(3, 9),  n_range=(4, 9), wmax=20, A_lo=0,  A_hi=3),
    'wider_weights': dict(B_range=(3, 9),  n_range=(4, 9), wmax=50, A_lo=0,  A_hi=3),
    # both
    'blocks_weights':dict(B_range=(8, 16), n_range=(4, 9), wmax=20, A_lo=0,  A_hi=3),
    # break nonnegativity of A (downward-closed feasibility no longer holds)
    'negative_A':    dict(B_range=(8, 16), n_range=(4, 9), wmax=20, A_lo=-2, A_hi=3,
                          neg_rhs_jitter=2.0),
    # the kitchen sink
    'stress':        dict(B_range=(8, 18), n_range=(4, 10), wmax=50, A_lo=-3, A_hi=4,
                          neg_rhs_jitter=3.0),
}
