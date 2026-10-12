# Block-separable binary programs: an exact hull-correction solver

My IE210 course project at IIT Bombay. It is an exact, self-certifying solver for 0/1 programs that split into small blocks tied together by one linear equation, plus a large-scale search for the instances that make it work hardest.

## What it is

The problem family (the general binary family from our professor) is

```
min c'x   s.t.   A_b x_b <= b_b   for every block b,    sum_i w_i x_i = S,    x in {0,1}^n
```

Each block has 8-13 binary variables and its own general constraints. A single coupling row ties the blocks together, and that row makes the problem NP-hard in general. The question I took on: how fast can this be solved *exactly*, and when does the easy convex structure break?

## What I did

1. **Exact reduction.** For each block I enumerate its feasible patterns once, which gives a table `T_b[s]` = cheapest cost at coupling-sum `s`. The whole problem then becomes exactly one value of a min-plus convolution `G[S] = (T_1 ⊕ ... ⊕ T_B)[S]`. `general_binary_theory_and_hunt.ipynb` checks this against brute force over the full `x`.
2. **Convex case (Theorems A and B in the notebook).** With unit weights, block tables are convex (the sum of the `s` cheapest items). Convex tables convolve as a global k-way merge of sorted hull increments in O(N log N), implemented with a Fenwick tree (`IncTree` in `v3_core.py`).
3. **The non-convex catch.** With real knapsack-style weights most block tables are non-convex, so the hull merge only gives a *lower bound*. The notebook notes that an earlier prototype returned this bound as if it were exact, and that this undershot about 38% of the time. Finding that bug is why everything afterwards is checked against exact convolution.
4. **Hull-correction + A\* solver (`v3_solver.py`).** I defined the *depth* `d(S)`: the minimum number of blocks that must sit strictly above their own lower hull in an optimal allocation. The exact value is the hull merge plus a correction over `d` deviating blocks. `solve_v3` searches d = 0..3, prunes candidates with Fenwick lower bounds, and computes exact values with windowed restricted convolutions. It certifies optimality when the incumbent beats `hullBound + (sum of the d+1 smallest distinct deviation gaps)`. The speed-ups in `v3_benchmark.py` are greedy drift correction, prefix/suffix DPs for leave-one-out queries, a clip-aware early exit, a restricted-hull tree and an integrality cut (`ceil(LB)` is valid when costs are integers).
5. **Benchmark vs Gurobi** (`general_binary_full.ipynb`, Part 1, 60 s cap). Over N = 100…10,000 my solver matched ground truth and certified in 24/24 cases. Against default Gurobi it was faster in most rows (e.g. 3.8x, 9.2x, 6.3x at N = 10,000). It was clearly **slower** where depth 2 was needed: 0.70 s vs 0.16 s at N = 1,000, and **108.9 s vs 1.48 s** at N = 5,000. "Raw" Gurobi (presolve/cuts off) hit the 60 s cap in 16 of the 24 rows. On my own solver alone, N = 20,000 (1,998 blocks) took 1.39 s for the coupling stage.
6. **Depth hunt.** `depth_hunter.py` uses exact convolution plus a polynomial DP over (coupling-sum, #deviating blocks), with no solver involved. A 2-hour run on the wide family covered **241,503,295** (instance, S) cases: d=0 87.53%, d=1 12.30%, d=2 0.18%, d=3 **568** cases, d≥4 none (`depth_hunt_overflow.json` is empty). The 568 depth-3 specimens are in `evil_instances.json` and were re-verified by an independent DP (568/568, 559 with a unique optimum).
7. **Stress outside the comfortable regime.** `gen_extended.py` widens the parameters (more blocks, weights up to 20/50, negative `A` entries). `exp_depth_extended.py` audits depth and solver exactness there. It found **6 depth-4 specimens** (`exp_depth_highdepth.json`), all with coupling weights up to ~50. So "d ≤ 3" is a property of the small-weight family, not of the problem. `verify_all.py` independently checks every primitive, including on constructed depth-4/5 gadgets.

## What I learnt

- **Reformulating mattered more than any coding trick.** Once I saw the problem as one min-plus convolution, the rest of the project (hulls, depth, certificates) followed from it.
- **A fast wrong answer is worse than a slow right one.** The prototype's hull bound looked great until I compared it with ground truth. Since then every speed-up was re-validated against exact convolution (the benchmark docstring records 12,543/12,543 after each change).
- **Averages hide the tail.** The solver is usually faster than Gurobi, but one depth-2 instance at N = 5,000 took 108.9 s where Gurobi needed 1.5 s. Depth, not N, is what drives the cost, and I should present results that way rather than as an average "speed-up".
- **Sampling evidence is not a theorem, and the conjecture broke when I widened the family.** 241M cases with max depth 3 felt convincing, yet spreading the weights to 50 produced depth-4 instances. The constructed gadgets in `verify_all.py` were how I made sure the detector could see d≥4 at all.
- **Verification code should share as little as possible with the code it checks.** `verify_all.py` uses bit-mask enumeration and raw-`x` brute force rather than calling the same functions again.
- **Unfinished:** the large-N timing sweep on the extended families (`exp_timing_extended.py`) never produced results (`exp_timing_results.csv` is empty). Proving or characterising a depth bound, and handling multiple coupling rows, are still open.

## Repo structure

```
v3_core.py                    instance generators, block tables, lower hulls, min-plus conv, Fenwick IncTree
v3_solver.py                  solve_v3: exact A* hull-correction solver with certificates
v3_benchmark.py               validate / sweep / gurobi modes
verify_all.py                 independent verification battery
depth_hunter.py               billion-scale exact depth probe (CLI)
gen_extended.py               parameterised generator for stress configs
exp_depth_extended.py         depth + solver-exactness audit on stress configs
exp_timing_extended.py        large-N timing sweep (no results yet)
general_binary_full.ipynb     solver, Gurobi benchmark, 2-hour hunt, specimen analysis (with outputs)
general_binary_theory_and_hunt.ipynb   theorems, proofs-by-experiment, hunt walkthrough
evil_instances.json           568 verified depth-3 instances
exp_depth_highdepth.json      6 depth-4 instances from the wide-weight configs
depth_hunt_overflow.json, exp_depth_wrong.json, exp_timing_results.csv   empty outputs (nothing found / not run)
```

## How to run

```bash
pip install numpy               # gurobipy only for the Gurobi comparison (needs a license)
python verify_all.py            # every line should end OK / 0 bad (~3-4 min)
python v3_benchmark.py validate # solver vs exact convolution
python v3_benchmark.py sweep    # timing N = 1000..20000
python depth_hunter.py --cases 1e7 --family wide
python exp_depth_extended.py --configs wide_weights negative_A --inst 2000
```

The notebooks expect the `.py` files in the same folder.
