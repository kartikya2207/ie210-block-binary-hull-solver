"""
v3 hull-correction solver for the general binary family
  min c'x  s.t.  A_b x_b <= b_b (per block),  sum_i w_i x_i = beta,  x in {0,1}^n

Pipeline:
  Level 1: per-block exact tables T_b[s] (vectorized enumeration, small blocks)
  Hull layer: lower convex hulls H_b, unit increments, global sorted merge
              -> hullBound(beta) = prefix sum of beta smallest increments (Theorem B)
  d=1: for every deviation (b, s): T_b[s] + M_{-b}(beta - s) via Fenwick delete/query/reinsert
       also records slack(b,s) = H_b[s] + M_{-b}(beta-s) - hullBound  (>= 0)
  d=2: pairs in increasing (delta1+delta2) via heap;
       prune by  hullBound + d1 + d2 + max(slack1, slack2) >= best ;
       survivors -> Fenwick leave-two-out query
  d=3: same idea with triples (only entered if uncertified after d=2)
  Certificate: best <= hullBound + sum of (d+1) smallest deltas from distinct blocks
"""
import numpy as np
from itertools import product
import heapq, time

EPS = 1e-9
INF = float('inf')

# ---------- instance generator: sir's general binary family (verbatim) ----------
def gen_instance(N, seed):
    rng = np.random.RandomState(seed)
    blocks = []
    i = 0
    while i < N:
        nb = int(min(rng.randint(8, 13), N - i))
        blocks.append(list(range(i, i + nb)))
        i += nb
    c = rng.randint(-10, 11, N).astype(float)
    w = rng.randint(1, 4, N)                       # coupling weights alpha_i in {1,2,3}
    As, bs = [], []
    for idx in blocks:
        nb = len(idx)
        A = rng.randint(0, 4, (2, nb)).astype(float)   # 2 general rows per block
        bvec = A.sum(axis=1) * rng.uniform(0.35, 0.7, 2)
        As.append(A); bs.append(bvec)
    return dict(N=N, c=c, w=w, blocks=blocks, As=As, bs=bs, seed=seed)

# wide family used by the evil hunter (1-3 rows, weights up to 5, varied tightness)
def gen_wide(seed, nb_range=(4, 9), wmax=5, rows=(1, 3)):
    rng = np.random.RandomState(seed)
    B = rng.randint(3, 9)
    blocks, cs, ws, As, bs = [], [], [], [], []
    i = 0
    for _ in range(B):
        nb = rng.randint(*nb_range)
        blocks.append(list(range(i, i + nb))); i += nb
        cs.append(rng.randint(-10, 11, nb).astype(float))
        ws.append(rng.randint(1, wmax + 1, nb))
        r = rng.randint(rows[0], rows[1] + 1)
        A = rng.randint(0, 4, (r, nb)).astype(float)
        bvec = A.sum(axis=1) * rng.uniform(0.25, 0.8, r)
        As.append(A); bs.append(bvec)
    return dict(N=i, c=np.concatenate(cs), w=np.concatenate(ws),
                blocks=blocks, As=As, bs=bs, seed=seed)

# ---------- Level 1 ----------
def block_table(c, A, bvec, w):
    n = len(c)
    bits = np.array(list(product([0, 1], repeat=n)), dtype=np.int8)
    feas = np.all(bits @ A.T <= bvec + 1e-12, axis=1)
    sv = bits[feas] @ w; cv = bits[feas] @ c
    T = np.full(int(w.sum()) + 1, np.inf)
    np.minimum.at(T, sv.astype(int), cv)
    return T

def lower_hull(T):
    pts = [(s, T[s]) for s in range(len(T)) if T[s] < 1e17]
    hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][1]-hull[-2][1])*(p[0]-hull[-1][0]) >= (p[1]-hull[-1][1])*(hull[-1][0]-hull[-2][0]):
            hull.pop()
        hull.append(p)
    H = np.full(len(T), np.inf)
    for (x1, y1), (x2, y2) in zip(hull[:-1], hull[1:]):
        sl = (y2-y1)/(x2-x1) if x2 > x1 else 0.0
        for s in range(x1, x2+1):
            H[s] = y1 + sl*(s-x1)
    if hull:
        H[hull[-1][0]] = hull[-1][1]
    return H

# ---------- exact ground truth ----------
def minplus(G, T):
    out = np.full(len(G)+len(T)-1, np.inf)
    for j in range(len(T)):
        if T[j] >= 1e17:
            continue
        np.minimum(out[j:j+len(G)], G + T[j], out=out[j:j+len(G)])
    return out

def conv_list(tables):
    G = np.array([0.0])
    for t in tables:
        G = minplus(G, t)
    return G

# ---------- Fenwick over globally sorted increments: (count, sum) ----------
class IncTree:
    """Order-statistics tree over the globally sorted unit-increment list.
    Supports: deactivate/activate all increments of a block, and
    query(m) = sum of the m smallest currently-active increments (exact m, else None)."""
    def __init__(self, hulls):
        ev = []      # (value, block)
        self.block_pos = [[] for _ in hulls]
        for b, H in enumerate(hulls):
            smax = int(np.max(np.where(H < 1e17)[0]))
            for s in range(smax):
                ev.append((H[s+1] - H[s], b))
        ev.sort(key=lambda t: t[0])
        self.n = len(ev)
        self.val = np.array([v for v, _ in ev])
        for pos, (_, b) in enumerate(ev):
            self.block_pos[b].append(pos)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.cnt = np.zeros(2*self.size, dtype=np.int64)
        self.sm = np.zeros(2*self.size)
        self.cnt[self.size:self.size+self.n] = 1
        self.sm[self.size:self.size+self.n] = self.val
        for i in range(self.size-1, 0, -1):
            self.cnt[i] = self.cnt[2*i] + self.cnt[2*i+1]
            self.sm[i] = self.sm[2*i] + self.sm[2*i+1]
        self.active_total = self.n

    def _set(self, pos, on):
        i = self.size + pos
        self.cnt[i] = 1 if on else 0
        self.sm[i] = self.val[pos] if on else 0.0
        i //= 2
        while i:
            self.cnt[i] = self.cnt[2*i] + self.cnt[2*i+1]
            self.sm[i] = self.sm[2*i] + self.sm[2*i+1]
            i //= 2

    def remove_block(self, b):
        for p in self.block_pos[b]:
            self._set(p, False)
        self.active_total -= len(self.block_pos[b])

    def add_block(self, b):
        for p in self.block_pos[b]:
            self._set(p, True)
        self.active_total += len(self.block_pos[b])

    def query(self, m):
        """sum of m smallest active increments; None if fewer than m active."""
        if m < 0 or m > self.active_total:
            return None
        if m == 0:
            return 0.0
        i, acc = 1, 0.0
        rem = m
        while i < self.size:
            lc = self.cnt[2*i]
            if lc >= rem:
                i = 2*i
            else:
                acc += self.sm[2*i]; rem -= lc; i = 2*i + 1
        return acc + (self.sm[i] if rem == 1 else 0.0)  # leaf cnt is 0/1; rem==1 here

# ---------- helper: k smallest deltas from distinct blocks ----------
def smallest_distinct(dev_sorted, k):
    """dev_sorted: list of (delta, b, s) sorted by delta. Returns sum of k smallest
    deltas using distinct blocks, or INF if impossible."""
    seen, tot = set(), 0.0
    for d, b, s in dev_sorted:
        if b in seen:
            continue
        seen.add(b); tot += d
        if len(seen) == k:
            return tot
    return INF

# ---------- the v3 solver ----------
def solve_hull_correction_v3(tabs, beta, max_depth=3, pair_cap=2_000_000):
    t0 = time.perf_counter()
    B = len(tabs)
    hulls = [lower_hull(T) for T in tabs]
    tree = IncTree(hulls)

    # deviations: achievable points strictly above the hull
    dev = []   # (delta, b, s)
    for b, (T, H) in enumerate(zip(tabs, hulls)):
        for s in range(len(T)):
            if T[s] < 1e17 and H[s] < 1e17 and T[s] - H[s] > 1e-9:
                dev.append((T[s] - H[s], b, s))
    dev.sort(key=lambda t: t[0])

    # ---- d = 0 ----
    hull_bound = tree.query(beta)
    if hull_bound is None:
        return dict(obj=INF, depth=-1, certified=False, time=time.perf_counter()-t0)
    best, depth_used = hull_bound, 0

    # certificate after d=0
    if best <= hull_bound + smallest_distinct(dev, 1) - 1e-9:
        return dict(obj=best, depth=0, certified=True, time=time.perf_counter()-t0,
                    hull_bound=hull_bound, n_dev=len(dev))

    # ---- d = 1 (full pass; also records slacks for the d=2 filter) ----
    slack = {}     # (b, s) -> slack value
    by_block = {}
    for d, b, s in dev:
        by_block.setdefault(b, []).append((d, s))
    for b, lst in by_block.items():
        tree.remove_block(b)
        for d, s in lst:
            r = tree.query(beta - s)
            if r is None:
                slack[(b, s)] = INF
                continue
            cand = tabs[b][s] + r
            slack[(b, s)] = (hulls[b][s] + r) - hull_bound
            if cand < best - 1e-12:
                best, depth_used = cand, 1
        tree.add_block(b)

    if best <= hull_bound + smallest_distinct(dev, 2) - 1e-9:
        return dict(obj=best, depth=depth_used, certified=True, time=time.perf_counter()-t0,
                    hull_bound=hull_bound, n_dev=len(dev), pairs_eval=0)

    # ---- d = 2: heap over pairs by delta1+delta2, slack filter, Fenwick survivors ----
    nD = len(dev)
    pairs_eval = pairs_seen = 0
    if nD >= 2 and max_depth >= 2:
        heap = [(dev[0][0] + dev[1][0], 0, 1)]
        seenh = {(0, 1)}
        cur_pair = None   # (b1, b2) currently removed from tree
        while heap:
            dsum, i, j = heapq.heappop(heap)
            if dsum >= best - hull_bound - 1e-12:
                break
            pairs_seen += 1
            if pairs_seen > pair_cap:
                break
            d1, b1, s1 = dev[i]; d2, b2, s2 = dev[j]
            if b1 != b2:
                if dsum + max(slack[(b1, s1)], slack[(b2, s2)]) < best - hull_bound - 1e-12:
                    if cur_pair != (b1, b2):
                        if cur_pair is not None:
                            tree.add_block(cur_pair[0]); tree.add_block(cur_pair[1])
                        tree.remove_block(b1); tree.remove_block(b2)
                        cur_pair = (b1, b2)
                    r = tree.query(beta - s1 - s2)
                    pairs_eval += 1
                    if r is not None:
                        cand = tabs[b1][s1] + tabs[b2][s2] + r
                        if cand < best - 1e-12:
                            best, depth_used = cand, 2
            # heap successors
            if j + 1 < nD and (i, j+1) not in seenh:
                heapq.heappush(heap, (dev[i][0] + dev[j+1][0], i, j+1)); seenh.add((i, j+1))
            if i + 1 < j and (i+1, j) not in seenh:
                heapq.heappush(heap, (dev[i+1][0] + dev[j][0], i+1, j)); seenh.add((i+1, j))
        if cur_pair is not None:
            tree.add_block(cur_pair[0]); tree.add_block(cur_pair[1])

    if best <= hull_bound + smallest_distinct(dev, 3) - 1e-9:
        return dict(obj=best, depth=depth_used, certified=True, time=time.perf_counter()-t0,
                    hull_bound=hull_bound, n_dev=len(dev), pairs_eval=pairs_eval)

    # ---- d = 3 (rare; evil instances) ----
    triples_eval = 0
    if nD >= 3 and max_depth >= 3:
        gap = best - hull_bound
        cands = [(d, b, s) for d, b, s in dev if d < gap - 1e-12]
        cands.sort(key=lambda t: t[0])
        for a in range(len(cands)):
            da, ba, sa = cands[a]
            if 3*da >= gap:    # sorted: da <= db <= dc
                break
            for bidx in range(a+1, len(cands)):
                db_, bb, sb = cands[bidx]
                if da + 2*db_ >= gap:
                    break
                if bb == ba:
                    continue
                for cidx in range(bidx+1, len(cands)):
                    dc, bc, sc = cands[cidx]
                    if da + db_ + dc >= best - hull_bound - 1e-12:
                        break
                    if bc == ba or bc == bb:
                        continue
                    ms = max(slack[(ba, sa)], slack[(bb, sb)], slack[(bc, sc)])
                    if da + db_ + dc + ms >= best - hull_bound - 1e-12:
                        continue
                    tree.remove_block(ba); tree.remove_block(bb); tree.remove_block(bc)
                    r = tree.query(beta - sa - sb - sc)
                    tree.add_block(ba); tree.add_block(bb); tree.add_block(bc)
                    triples_eval += 1
                    if r is not None:
                        cand = tabs[ba][sa] + tabs[bb][sb] + tabs[bc][sc] + r
                        if cand < best - 1e-12:
                            best, depth_used = cand, 3
                            gap = best - hull_bound

    certified = best <= hull_bound + smallest_distinct(dev, 4) - 1e-9
    return dict(obj=best, depth=depth_used, certified=certified, time=time.perf_counter()-t0,
                hull_bound=hull_bound, n_dev=len(dev), pairs_eval=pairs_eval,
                triples_eval=triples_eval)
