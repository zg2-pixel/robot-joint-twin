"""The twin as an OODA loop, one pass per cycle. Observe = health checks on the cycle, Orient = physics fit +
reference tracking + residual test + classifier + forecast, Decide = expected-cost rule, Act = speed override /
ticket / log. Works the same on recorded cycles and on simulator cycles."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from .features import cycle_features
from .physics import cycle_design, fit_friction
from .bayes import NIG, ReferenceTracker
from .detect import HealthCheck, ResidualTest, FEATURES
from .forecast import RestartingTrendKF, summarize_crossings
from .decide import DecisionRule


class OODATwin:
    def __init__(self, ref_table, classifier, tau_max, lam=0.98, k=4.0, kf_kwargs=None, costs=None, window=30, seed=0,
                 trusted=("payload", "collision", "electrical")):
        """ref_table: per-cycle table of healthy reference cycles (features + fits) used for the bands and the prior.
        classifier: fitted, calibrated classifier over FEATURES with a 'healthy' class.
        trusted: the non-wear classes the decision may act on (a class the classifier cannot really see at this
        joint, e.g. light collisions at the shoulder, only adds noise and is left to the residual test)."""
        self.k = k
        self.tau_max = tau_max
        self.health = HealthCheck(ref_table, k=k)
        self.resid = ResidualTest(ref_table, k=k)
        self.clf = classifier
        self.classes = list(classifier.classes_)
        self.trusted = set(trusted)
        self.spread = {c: float(ref_table[c].std()) for c in ["load_coulomb", "load_c0"]}
        self.ref_tc0 = float(ref_table["load_coulomb"].mean())
        self.tracker = None                      # built from the first cycles
        self.lam = lam
        self.kf = RestartingTrendKF(level0=self.ref_tc0, ref_mean=self.ref_tc0, ref_std=self.spread["load_coulomb"], k_onset=1.5,
                                    **(kf_kwargs or {"obs_std": self.spread["load_coulomb"], "q_level": 0.01, "q_slope": 1e-4}))
        self.rule = DecisionRule(costs=costs, lead=window)
        self.rng = np.random.default_rng(seed)
        self.warm = []
        self.p_other = 0.0
        self.jump_votes = 0
        self.speed = 1.0
        self.ticket = None
        self.ticket_doubt = 0
        self.log = []

    # ---------------------------------------------------------------- the four stages
    def observe(self, frame, speed_applied=1.0):
        f = cycle_features(frame)
        f["duration"] = float(frame["time"].max()) * speed_applied   # a cycle run at 70 % speed is longer by design: compare at nominal speed
        flags = self.health(f)
        return f, flags

    def orient(self, frame, f):
        X, yl, yg = cycle_design(frame)
        fl = fit_friction(yl, X[:, 2], X[:, 3]); fg = fit_friction(yg, X[:, 2], X[:, 3])
        bl = np.array([fl["c0"], fl["coulomb"], fl["viscous"], fl["inertia"]])
        bg = np.array([fg["c0"], fg["coulomb"], fg["viscous"], fg["inertia"]])
        if self.tracker is None:
            self.warm.append((bl, bg, fl["noise"], fg["noise"]))
            if len(self.warm) >= 20:
                self.tracker = ReferenceTracker(NIG.from_cycles([w[0] for w in self.warm], [w[2] for w in self.warm]),
                                                NIG.from_cycles([w[1] for w in self.warm], [w[3] for w in self.warm]), lam=self.lam)
            z_tc = 0.0; z_c0 = 0.0; ref_tc = self.ref_tc0; ref_std = self.spread["load_coulomb"]
        else:
            ref = self.tracker.load
            ref_std = ref.param_std()[1]
            z_tc = (bl[1] - ref.m[1]) / np.sqrt(self.spread["load_coulomb"] ** 2 + ref_std ** 2)
            z_c0 = (bl[0] - ref.m[0]) / np.sqrt(self.spread["load_c0"] ** 2 + ref.param_std()[0] ** 2)
            ref_tc = ref.m[1]
        z_res = self.resid.z(f)
        probs = dict(zip(self.classes, self.clf.predict_proba(pd.DataFrame([f[FEATURES]]))[0]))
        p_fault_clf = 1.0 - probs.get("healthy", 0.0)
        # acute risk, three sources: (1) the joint is beyond the limit right now (posterior of the Kalman level),
        # (2) a fault that is not wear - collision, electrical, and a payload change at a lower weight because it
        # is a task change rather than damage, (3) an unexplained residual = fault of unknown type
        w = {"collision": 1.0, "electrical": 1.0, "payload": 0.3}
        p_other_raw = sum(w[c] * probs.get(c, 0.0) for c in w if c in self.trusted)
        self.p_other = 0.5 * self.p_other + 0.5 * p_other_raw      # two-cycle memory: one noisy cycle should not slow the line
        p_other = self.p_other
        p_unknown = 1.0 if z_res > self.k else 0.0
        # the reference only learns from cycles that look healthy
        healthy_now = (abs(z_tc) < self.k) and (abs(z_c0) < self.k) and (z_res <= self.k)
        if self.tracker is not None:
            self.tracker.step(X, yl, yg, healthy=healthy_now)
        # forecast on the per-cycle tau_c, and the probability of being past the limit already. The trend filter
        # follows the reference while the joint is healthy and restarts when tau_c leaves the healthy band; a big
        # surprise (innovation) twice in a row is a step and resets the level to what was just seen.
        self.kf.update_reference(ref_tc, self.spread["load_coulomb"])
        innov = self.kf.step(bl[1])
        self.jump_votes = self.jump_votes + 1 if (innov == innov and abs(innov) > 3) else 0
        if self.jump_votes >= 2:                                 # two surprises in a row: a step, not an outlier
            self.kf.x[0] = bl[1]
            self.kf.P[0, 0] = self.spread["load_coulomb"] ** 2
        p_over = float(norm.sf((self.tau_max - self.kf.level) / max(self.kf.level_std(), 1e-6)))
        p_fault = max(p_over, min(p_other, 1.0), p_unknown)
        cross = self.kf.cycles_to_limit(self.tau_max, horizon=600, n=800, rng=self.rng)
        fc = summarize_crossings(cross, 600)
        p_window = float((cross <= self.rule.W).mean())
        t_plan = max(fc["lo5"], self.rule.W)                    # a planned stop needs the lead time W to organise
        p_before_lower = float((cross <= t_plan).mean())
        p_plan_horizon = float((cross <= self.rule.W + self.rule.plan_margin).mean())
        unused = max(fc["median"] - t_plan, 0.0)
        self._plan = (t_plan, unused, p_plan_horizon)
        return {"tau_c": bl[1], "c0": bl[0], "b_load": bl[2], "gear_b": bg[2], "ref_tc": ref_tc, "ref_tc_std": ref_std,
                "z_tc": z_tc, "z_c0": z_c0, "z_res": z_res, "p_fault": p_fault, "p_fault_clf": p_fault_clf,
                "p_over_limit": p_over, "p_other_fault": p_other, "kf_innovation": float(innov) if innov == innov else np.nan,
                "unknown_flag": bool(p_unknown), "kf_level": self.kf.level, "kf_slope": self.kf.slope,
                "n_median": fc["median"], "n_lo5": fc["lo5"], "n_hi95": fc["hi95"], "p_cross_window": p_window,
                "t_plan": t_plan, "p_cross_plan": p_before_lower, "p_cross_plan_horizon": p_plan_horizon,
                **{f"p_{c}": v for c, v in probs.items()}}, p_before_lower

    def decide(self, o, flags, p_before_lower):
        sensor_ok = not any(flags.values())
        # no ticket before the reference exists: the first forecasts are wide and would plan on nothing
        p_plan = self._plan[2] if self.tracker is not None else 0.0
        action, ec, note = self.rule.decide(o["p_fault"], o["p_cross_window"], p_before_lower, sensor_ok=sensor_ok,
                                            unused_life=self._plan[1], lo5=o["n_lo5"], ticket_open=self.ticket is not None,
                                            p_cross_plan_horizon=p_plan)
        return action, ec, note

    def act(self, action, o):
        """Speed override and ticket, plus the record for the log/dashboard."""
        self.speed = 0.7 if action == "slow" else 1.0
        note = ""
        if action == "schedule" and self.ticket is None:
            self.ticket = {"issued_at_cycle": len(self.log), "planned_in": o["t_plan"], "window_lo": o["n_lo5"], "window_hi": o["n_hi95"], "median": o["n_median"]}
            self.ticket_doubt = 0
        elif self.ticket is not None:
            # a ticket the forecast has stopped supporting for 10 cycles is withdrawn (it was a false start)
            self.ticket_doubt = self.ticket_doubt + 1 if self._plan[2] < self.rule.p_commit / 3 else 0
            if self.ticket_doubt >= 10:
                self.ticket = None
                note = "ticket withdrawn: the forecast no longer supports it"
        return {"speed": self.speed, "ticket_open": self.ticket is not None}, note

    # ---------------------------------------------------------------- one pass
    def step(self, frame, truth=None, speed_applied=1.0):
        """speed_applied: the speed override the frame was actually produced with (1.0 for recorded cycles,
        which cannot be slowed down after the fact)."""
        f, flags = self.observe(frame, speed_applied)
        o, p_before_lower = self.orient(frame, f)
        action, ec, note = self.decide(o, flags, p_before_lower)
        a, act_note = self.act(action, o)
        note = "; ".join(s for s in [note, act_note] if s)
        rec = {"cycle": len(self.log), "action": action, "note": note, "sensor_ok": not any(flags.values()),
               **{f"flag_{k}": v for k, v in flags.items()}, **o, **{f"cost_{k}": v for k, v in ec.items()}, **a}
        if truth is not None:
            rec.update(truth)
        self.log.append(rec)
        return rec

    def frame(self):
        return pd.DataFrame(self.log)
