"""
v3: exact hull-correction solver, A* architecture.

Layers (all EXACT by construction):
  value(F) = sum_{b in F} T_b[s_b]  +  RC_{-F}(beta - sum s_b)
  where RC_X(m) = exact min-plus convolution of the ON-HULL RESTRICTED tables
  R_b[s] = T_b[s] if T_b[s] == H_b[s] else +inf, over blocks X, at exactly m.

Speed comes from never computing RC arrays:
  * Fenwick greedy hull merge gives valid LOWER bounds for every candidate
    (pruning + delta-sum certificates), in O(log) per query.
  * Only candidates whose LB beats the incumbent get an EXACT RC point query,
    done as a windowed restricted convolution around the greedy trajectory,
    verified against the LB and escalated (window doubling -> full) if needed.
"""
import numpy as np
import heapq, time
from v3_core import gen_instance, gen_wide, block_table, lower_hull, minplus, conv_list, IncTree

EPS = 1e-9
INF = float('inf')


def build_structures(tabs):
    hulls, Rs, smaxs = [], [], []
    dev = []                      # (delta, b, s)
    for b, T in enumerate(tabs):
        H = lower_hull(T)
        hulls.append(H)
        fin = np.where(H < 1e17)[0]
        smaxs.append(int(fin.max()) if len(fin) else 0)
        R = np.full(len(T), np.inf)
        with np.errstate(invalid='ignore'):
            on = (T < 1e17) & (H < 1e17) & (T - H <= 1e-9)
        R[on] = T[on]
        Rs.append(R)
        ach = np.where((T < 1e17) & (H < 1e17))[0]
        for s in ach:
            d = T[s] - H[s]
            if d > 1e-9:
                dev.append((float(d), b, int(s)))
    dev.sort(key=lambda t: t[0])
    return hulls, Rs, smaxs, dev


class ExactEvaluator:
    """Exact RC_X(m) point queries via windowed restricted convolution.

    Window is centered on the greedy hull trajectory (per-block positions from
    the global slope threshold). Result is verified: if it equals the Fenwick
    LB it is exact; otherwise the window doubles, escalating to the full
    m-capped restricted convolution which is exact unconditionally.
    """
    def __init__(self, tabs, Rs, tree, smaxs, beta=None):
        self.tabs, self.Rs, self.tree, self.smaxs = tabs, Rs, tree, smaxs
        self.B = len(tabs)
        # per-block sorted increment values for greedy positions
        self.inc_vals = [tree.val[tree.block_pos[b]] for b in range(self.B)]
        # cached restricted domains: (achievable on-hull s values, their costs)
        self.dom = []
        for R in Rs:
            sd = np.where(R < 1e17)[0]
            self.dom.append((sd, R[sd].copy()))
        self.calls = 0; self.escalations = 0
        # verification tree over hulls of the RESTRICTED tables: a tighter valid
        # lower bound on RC than the full-table hull tree (closes most queries).
        self.tree2 = IncTree([lower_hull(R) for R in Rs])
        # integral costs => RC values are integers => LBs may be rounded up (sound)
        self.integral = all(len(rc) == 0 or np.allclose(rc, np.round(rc), atol=1e-9)
                            for _, rc in self.dom)
        # prefix/suffix windowed DPs around the full-set greedy trajectory:
        # answers |excl|<=1 queries in O(W) instead of an O(B*W) pass.
        self.ps = None
        if beta is not None and 0 < beta <= tree.active_total:
            posF, totF = self._greedy_positions(frozenset(), beta)
            if totF == beta:
                self._build_ps(posF)

    def _build_ps(self, posF, Wps=32):
        dom, B = self.dom, self.B
        P = [None]*(B+1); bP = [0]*(B+1); P[0] = np.array([0.0])
        traj = 0
        for b in range(B):
            sdom, rcost = dom[b]; traj += int(posF[b])
            lo = max(0, traj - Wps); hi = traj + Wps
            nd = np.full(hi - lo + 1, np.inf)
            dp, base = P[b], bP[b]
            for s, rc in zip(sdom, rcost):
                j0 = max(lo - s, base); j1 = min(hi - s, base + len(dp) - 1)
                if j0 > j1: continue
                seg = dp[j0-base:j1-base+1] + rc
                o0 = j0 + s - lo
                np.minimum(nd[o0:o0+(j1-j0+1)], seg, out=nd[o0:o0+(j1-j0+1)])
            P[b+1] = nd; bP[b+1] = lo
        self.ps = (P, bP, None, None)   # suffix built lazily
        self._posF, self._Wps = posF, Wps

    def _build_suffix(self):
        dom, B, Wps, posF = self.dom, self.B, self._Wps, self._posF
        S = [None]*(B+1); bS = [0]*(B+1); S[B] = np.array([0.0])
        traj = 0
        for b in range(B-1, -1, -1):
            sdom, rcost = dom[b]; traj += int(posF[b])
            lo = max(0, traj - Wps); hi = traj + Wps
            nd = np.full(hi - lo + 1, np.inf)
            dp, base = S[b+1], bS[b+1]
            for s, rc in zip(sdom, rcost):
                j0 = max(lo - s, base); j1 = min(hi - s, base + len(dp) - 1)
                if j0 > j1: continue
                seg = dp[j0-base:j1-base+1] + rc
                o0 = j0 + s - lo
                np.minimum(nd[o0:o0+(j1-j0+1)], seg, out=nd[o0:o0+(j1-j0+1)])
            S[b] = nd; bS[b] = lo
        P, bP, _, _ = self.ps
        self.ps = (P, bP, S, bS)

    def _ps_query(self, excl, m):
        """O(W) candidate value for |excl|<=1 from prefix/suffix DPs (upper bound on RC)."""
        P, bP, S, bS = self.ps
        if not excl:
            dp, base = P[self.B], bP[self.B]
            i = m - base
            return float(dp[i]) if 0 <= i < len(dp) else None
        if S is None:
            self._build_suffix()
            P, bP, S, bS = self.ps
        b = next(iter(excl))
        Pk, bp = P[b], bP[b]; Sk, bs_ = S[b+1], bS[b+1]
        j0 = max(bp, m - bs_ - len(Sk) + 1); j1 = min(bp + len(Pk) - 1, m - bs_)
        if j0 > j1:
            return None
        a = Pk[j0-bp:j1-bp+1]
        c = Sk[m-j1-bs_:m-j0-bs_+1][::-1]
        return float(np.min(a + c))

    def _greedy_positions(self, excl, m):
        """approximate greedy split of m over blocks not in excl"""
        # threshold value = m-th smallest active increment (tree currently has excl removed)
        t = self.tree
        if m <= 0:
            return np.zeros(self.B, dtype=np.int64)
        i, rem = 1, min(m, t.active_total)
        while i < t.size:
            lc = t.cnt[2*i]
            if lc >= rem: i = 2*i
            else: rem -= lc; i = 2*i + 1
        lam = t.val[i - t.size] if i - t.size < t.n else t.val[-1]
        pos = np.zeros(self.B, dtype=np.int64)
        for b in range(self.B):
            if b in excl: continue
            pos[b] = np.searchsorted(self.inc_vals[b], lam, side='left')
        # exact drift correction: distribute residual over lambda-tied increments
        tot = int(pos.sum())
        extra = min(m, t.active_total) - tot
        if extra > 0:
            for b in range(self.B):
                if b in excl: continue
                avail = int(np.searchsorted(self.inc_vals[b], lam, side='right')) - int(pos[b])
                if avail > 0:
                    take = min(avail, extra)
                    pos[b] += take; extra -= take
                    if extra == 0: break
            tot = int(pos.sum())
        return pos, tot

    def query(self, excl, m, lb=None):
        """exact RC over blocks not in excl, at exactly m. excl: set of block ids.
        Assumes self.tree currently has exactly the excl blocks removed (for LB/greedy).
        lb: known Fenwick lower bound for verification (optional)."""
        self.calls += 1
        if m < 0:
            return INF
        if m == 0:
            base = 0.0
            for b in range(self.B):
                if b not in excl:
                    if self.Rs[b][0] >= 1e17: return INF  # s=0 always achievable in this family, defensive
                    base += self.Rs[b][0]
            return base
        if lb is None:
            lb = self.tree.query(m)
            if lb is None:
                lb = INF
        # tighter LB from the restricted-hull tree (sync excl, query, restore)
        t2 = self.tree2
        for b in excl: t2.remove_block(b)
        lb2 = t2.query(m)
        for b in excl: t2.add_block(b)
        if lb2 is None:
            return INF            # m exceeds max achievable restricted total -> RC = INF exactly
        if lb < INF:
            lb = max(lb, lb2)
        else:
            lb = lb2
        if self.integral and lb < INF:
            lb = float(np.ceil(lb - 1e-9))   # RC is integer-valued: round LB up (sound)
        # fast path: prefix/suffix combine, exact when it meets its own lower bound
        if self.ps is not None and len(excl) <= 1:
            v = self._ps_query(excl, m)
            if v is not None and v < INF and lb < INF and abs(v - lb) <= 1e-9:
                return v
        pos, tot = self._greedy_positions(excl, m)
        order = [b for b in range(self.B) if b not in excl]
        W = 16
        while True:
            v, clipped = self._windowed(order, m, pos, W)
            if not clipped:
                return v          # window never cut a reachable state -> full conv -> exact
            if lb < INF and abs(v - lb) <= 1e-9:
                return v          # matches its own lower bound -> exact
            if W > m:             # full conv was just performed -> exact
                return v
            W *= 4
            self.escalations += 1

    def _windowed(self, order, m, pos, W):
        """restricted convolution keeping only states within +-W of greedy trajectory.
        Returns (value, clipped): clipped=False means no reachable total below the
        m-cap was discarded, so the result equals the full restricted convolution."""
        dom = self.dom
        traj = 0
        dp = np.array([0.0]); base = 0     # dp[k] = cost at total (base + k)
        clipped = False
        for b in order:
            sdom, rcost = dom[b]
            if len(sdom) == 0:
                return INF, False   # some block has no achievable on-hull state: RC = INF exactly
            traj += int(pos[b])
            lo = max(0, traj - W); hi = min(m, traj + W)
            if lo > hi:
                return INF, True
            # reachable new totals: [base+min_s, base+len(dp)-1+max_s] (cap above m lossless)
            r_lo = base + int(sdom[0]); r_hi = min(m, base + len(dp) - 1 + int(sdom[-1]))
            if lo > r_lo or hi < r_hi:
                clipped = True
            L = hi - lo + 1
            nd = np.full(L, np.inf)
            for s, rc in zip(sdom, rcost):
                # old totals j = (new total) - s must lie in [base, base+len(dp)-1]
                j0 = max(lo - s, base); j1 = min(hi - s, base + len(dp) - 1)
                if j0 > j1: continue
                seg = dp[j0-base:j1-base+1] + rc
                o0 = j0 + s - lo
                np.minimum(nd[o0:o0+(j1-j0+1)], seg, out=nd[o0:o0+(j1-j0+1)])
            dp, base = nd, lo
        idx = m - base
        if 0 <= idx < len(dp):
            return float(dp[idx]), clipped
        return INF, clipped


def smallest_distinct(dev, k):
    seen, tot = set(), 0.0
    for d, b, s in dev:
        if b in seen: continue
        seen.add(b); tot += d
        if len(seen) == k:
            return tot
    return INF


def solve_v3(tabs, beta, max_depth=3, pair_cap=500_000):
    t0 = time.perf_counter()
    B = len(tabs)
    hulls, Rs, smaxs, dev = build_structures(tabs)
    tree = IncTree(hulls)
    ev = ExactEvaluator(tabs, Rs, tree, smaxs, beta=beta)
    stats = dict(n_dev=len(dev))

    LB0 = tree.query(beta)
    if LB0 is None:
        return dict(obj=INF, depth=-1, certified=False, time=time.perf_counter()-t0, **stats)

    # ---------- candidate gathering: d=0 and all d=1, with Fenwick LBs ----------
    cands = [(LB0, 0, None)]                  # (LBvalue, depth, payload)
    slack = {}
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
            slack[(b, s)] = (hulls[b][s] + r) - LB0
            cands.append((tabs[b][s] + r, 1, (b, s)))
        tree.add_block(b)

    # ---------- A* phase over d<=1 ----------
    cands.sort(key=lambda t: t[0])
    E, depth_used = INF, 0
    for lbv, d, pay in cands:
        if lbv >= E - 1e-12:
            break
        if d == 0:
            v = ev.query(frozenset(), beta, lb=LB0)
            if v < E - 1e-12: E, depth_used = v, 0
        else:
            b, s = pay
            tree.remove_block(b)
            v = tabs[b][s] + ev.query({b}, beta - s, lb=None)
            tree.add_block(b)
            if v < E - 1e-12: E, depth_used = v, 1

    if E <= LB0 + smallest_distinct(dev, 2) - 1e-9:
        return dict(obj=E, depth=depth_used, certified=True, time=time.perf_counter()-t0,
                    exact_evals=ev.calls, pairs_eval=0, **stats)

    # ---------- d=2: heap by delta-sum, slack filter, lazy exact eval ----------
    nD = len(dev); pairs_eval = 0
    if nD >= 2 and max_depth >= 2:
        heap = [(dev[0][0] + dev[1][0], 0, 1)]
        seen = {(0, 1)}
        while heap:
            dsum, i, j = heapq.heappop(heap)
            if dsum >= E - LB0 - 1e-12:
                break
            d1, b1, s1 = dev[i]; d2, b2, s2 = dev[j]
            if b1 != b2 and dsum + max(slack[(b1, s1)], slack[(b2, s2)]) < E - LB0 - 1e-12:
                m = beta - s1 - s2
                tree.remove_block(b1); tree.remove_block(b2)
                rlb = tree.query(m)
                if rlb is not None and tabs[b1][s1] + tabs[b2][s2] + rlb < E - 1e-12:
                    v = tabs[b1][s1] + tabs[b2][s2] + ev.query({b1, b2}, m, lb=rlb)
                    pairs_eval += 1
                    if v < E - 1e-12:
                        E, depth_used = v, 2
                tree.add_block(b1); tree.add_block(b2)
            if j + 1 < nD and (i, j+1) not in seen:
                heapq.heappush(heap, (dev[i][0] + dev[j+1][0], i, j+1)); seen.add((i, j+1))
            if i + 1 < j and (i+1, j) not in seen:
                heapq.heappush(heap, (dev[i+1][0] + dev[j][0], i+1, j)); seen.add((i+1, j))
            if pairs_eval > pair_cap:
                break

    if E <= LB0 + smallest_distinct(dev, 3) - 1e-9:
        return dict(obj=E, depth=depth_used, certified=True, time=time.perf_counter()-t0,
                    exact_evals=ev.calls, pairs_eval=pairs_eval, **stats)

    # ---------- d=3 (evil instances only) ----------
    triples_eval = 0
    if nD >= 3 and max_depth >= 3:
        gap = E - LB0
        cd = [(d, b, s) for d, b, s in dev if d < gap]
        for a in range(len(cd)):
            da, ba, sa = cd[a]
            if 3*da >= E - LB0 - 1e-12: break
            for bi in range(a+1, len(cd)):
                db_, bb, sb = cd[bi]
                if da + 2*db_ >= E - LB0 - 1e-12: break
                if bb == ba: continue
                for ci in range(bi+1, len(cd)):
                    dc, bc, sc = cd[ci]
                    if da + db_ + dc >= E - LB0 - 1e-12: break
                    if bc == ba or bc == bb: continue
                    ms = max(slack[(ba, sa)], slack[(bb, sb)], slack[(bc, sc)])
                    if da + db_ + dc + ms >= E - LB0 - 1e-12: continue
                    m = beta - sa - sb - sc
                    tree.remove_block(ba); tree.remove_block(bb); tree.remove_block(bc)
                    rlb = tree.query(m)
                    if rlb is not None and tabs[ba][sa]+tabs[bb][sb]+tabs[bc][sc]+rlb < E - 1e-12:
                        v = tabs[ba][sa]+tabs[bb][sb]+tabs[bc][sc] + ev.query({ba, bb, bc}, m, lb=rlb)
                        triples_eval += 1
                        if v < E - 1e-12:
                            E, depth_used = v, 3
                    tree.add_block(ba); tree.add_block(bb); tree.add_block(bc)

    certified = E <= LB0 + smallest_distinct(dev, 4) - 1e-9
    return dict(obj=E, depth=depth_used, certified=certified, time=time.perf_counter()-t0,
                exact_evals=ev.calls, pairs_eval=pairs_eval, triples_eval=triples_eval, **stats)
