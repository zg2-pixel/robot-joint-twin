"""The twin as an OODA loop, one pass per cycle. Observe = health checks on the cycle, Orient = physics fit +
reference tracking + residual test + classifier + forecast, Decide = expected-cost rule, Act = speed override /
ticket / log. Works the same on recorded cycles, feature-table rows, and simulator cycles."""
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
                 trusted=("payload", "collision", "electrical"), delta_tau_max=None, severe_z=None,
                 adaptive_health=False, gate_forecast=False):
        """ref_table: per-cycle table of healthy reference cycles (features + fits) used for the bands and the prior.
        classifier: fitted, calibrated classifier over FEATURES with a 'healthy' class.
        trusted: the non-wear classes the decision may act on (a class the classifier cannot really see at this
        joint, e.g. light collisions at the shoulder, only adds noise and is left to the residual test).
        delta_tau_max: if set, the wear limit is tracked relative to the healthy reference (ref_tc + delta_tau_max)
        instead of a fixed absolute tau_max, consistent with Deliverable 2 (only changes in tau_c are physical).
        severe_z: if set (e.g. 6.0), a single-cycle z-score above severe_z overrides the 2-cycle stop hysteresis.
        adaptive_health: if True, HealthCheck tracks the batch-level A-B offset with its within-batch spread.
        gate_forecast: if True, maintenance scheduling is enabled only once the trend filter's slope is confirmed.
        """
        self.k = k
        self.tau_max = tau_max
        self.delta_tau_max = delta_tau_max
        self.severe_z = severe_z
        self.gate_forecast = gate_forecast
        self.health = HealthCheck(ref_table, k=k, adaptive=adaptive_health, lam=lam)
        self.resid = ResidualTest(ref_table, k=k)
        self.clf = classifier
        self.classes = list(classifier.classes_)
        self.trusted = set(trusted)
        self.spread = {c: float(ref_table[c].std()) for c in ["load_coulomb", "load_c0"]}
        self.ref_tc0 = float(ref_table["load_coulomb"].mean())
        self.ref_c00 = float(ref_table["load_c0"].mean())
        self.tracker = None                      # built from the first cycles when raw frames are available
        self.lam = lam
        self._feat_ref_tc = self.ref_tc0
        self._feat_ref_c0 = self.ref_c00
        self._feat_ref_std = self.spread["load_coulomb"] / np.sqrt(max(len(ref_table), 1))
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

    def _effective_limit(self, ref_tc):
        if self.delta_tau_max is not None:
            return float(ref_tc + self.delta_tau_max)
        return float(self.tau_max)

    # ---------------------------------------------------------------- the four stages
    def observe(self, frame, speed_applied=1.0):
        f = cycle_features(frame)
        f["duration"] = float(frame["time"].max()) * speed_applied   # a cycle run at 70 % speed is longer by design: compare at nominal speed
        flags = self.health(f)
        return f, flags

    def _orient_core(self, f, bl, bg, ref_tc, ref_std, z_tc, z_c0, use_forecast=True, probs=None):
        z_res = self.resid.z(f)
        if probs is None:
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
        healthy_now = (abs(z_tc) < self.k) and (abs(z_c0) < self.k) and (z_res <= self.k)

        eff_limit = self._effective_limit(ref_tc)
        self.kf.update_reference(ref_tc, self.spread["load_coulomb"])
        innov = self.kf.step(bl[1])
        self.jump_votes = self.jump_votes + 1 if (innov == innov and abs(innov) > 3) else 0
        severe = (self.severe_z is not None) and ((abs(z_tc) >= self.severe_z) or (z_res >= self.severe_z))
        if self.jump_votes >= 2 or severe:
            self.kf.x[0] = bl[1]
            self.kf.P[0, 0] = self.spread["load_coulomb"] ** 2
        p_over = float(norm.sf((eff_limit - self.kf.level) / max(self.kf.level_std(), 1e-6)))
        p_fault = max(p_over, min(p_other, 1.0), p_unknown)

        if use_forecast:
            cross = self.kf.cycles_to_limit(eff_limit, horizon=600, n=800, rng=self.rng)
            fc = summarize_crossings(cross, 600)
            p_window = float((cross <= self.rule.W).mean())
            t_plan = max(fc["lo5"], self.rule.W)
            p_before_lower = float((cross <= t_plan).mean())
            p_plan_horizon = float((cross <= self.rule.W + self.rule.plan_margin).mean())
            unused = max(fc["median"] - t_plan, 0.0)
        else:
            fc = {"median": np.nan, "lo5": np.nan, "hi95": np.nan}
            p_window = p_over
            t_plan = float(self.rule.W)
            p_before_lower = 0.0
            p_plan_horizon = 0.0
            unused = 0.0

        forecast_ready = self.kf.is_ready() if self.gate_forecast else True
        if not forecast_ready:
            p_plan_horizon = 0.0
        self._plan = (t_plan, unused, p_plan_horizon)
        self._severe = severe

        o = {"tau_c": bl[1], "c0": bl[0], "b_load": bl[2], "gear_b": bg[2], "ref_tc": ref_tc, "ref_tc_std": ref_std,
             "eff_limit": eff_limit, "z_tc": z_tc, "z_c0": z_c0, "z_res": z_res, "p_fault": p_fault,
             "p_fault_clf": p_fault_clf, "p_over_limit": p_over, "p_other_fault": p_other,
             "kf_innovation": float(innov) if innov == innov else np.nan, "unknown_flag": bool(p_unknown),
             "kf_level": self.kf.level, "kf_slope": self.kf.slope, "forecast_ready": bool(forecast_ready),
             "severe_override": bool(severe), "n_median": fc["median"], "n_lo5": fc["lo5"], "n_hi95": fc["hi95"],
             "p_cross_window": p_window, "t_plan": t_plan, "p_cross_plan": p_before_lower,
             "p_cross_plan_horizon": p_plan_horizon, **{f"p_{c}": v for c, v in probs.items()}}
        return o, p_before_lower, healthy_now

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
        o, p_before_lower, healthy_now = self._orient_core(f, bl, bg, ref_tc, ref_std, z_tc, z_c0, use_forecast=True)
        if self.tracker is not None:
            self.tracker.step(X, yl, yg, healthy=healthy_now)
        return o, p_before_lower, healthy_now

    def orient_from_features(self, f, use_forecast=False, probs=None):
        """Orient directly from a pre-computed per-cycle feature row (e.g. cycles_axis2.csv)."""
        bl = np.array([float(f["load_c0"]), float(f["load_coulomb"]), float(f["load_viscous"]), float(f.get("load_inertia", 0.0))])
        bg = np.array([float(f.get("gear_c0", 0.0)), float(f.get("gear_coulomb", 0.0)), float(f.get("gear_viscous", 0.0)), float(f.get("gear_inertia", 0.0))])
        ref_tc = self._feat_ref_tc
        ref_std = self._feat_ref_std
        z_tc = (bl[1] - ref_tc) / np.sqrt(self.spread["load_coulomb"] ** 2 + ref_std ** 2)
        z_c0 = (bl[0] - self._feat_ref_c0) / self.spread["load_c0"]
        o, p_before_lower, healthy_now = self._orient_core(f, bl, bg, ref_tc, ref_std, z_tc, z_c0, use_forecast=use_forecast, probs=probs)
        if healthy_now:
            self._feat_ref_tc = self.lam * self._feat_ref_tc + (1.0 - self.lam) * bl[1]
            self._feat_ref_c0 = self.lam * self._feat_ref_c0 + (1.0 - self.lam) * bl[0]
        return o, p_before_lower, healthy_now

    def decide(self, o, flags, p_before_lower, allow_plan=None):
        sensor_ok = not any(flags.values())
        if allow_plan is None:
            allow_plan = self.tracker is not None
        p_plan = self._plan[2] if allow_plan else 0.0
        action, ec, note = self.rule.decide(o["p_fault"], o["p_cross_window"], p_before_lower, sensor_ok=sensor_ok,
                                            unused_life=self._plan[1], lo5=o["n_lo5"], ticket_open=self.ticket is not None,
                                            p_cross_plan_horizon=p_plan, severe=getattr(self, "_severe", False))
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
        sensor_ok = not any(flags.values())
        o, p_before_lower, healthy_now = self.orient(frame, f)
        self.health.update(f, healthy=(sensor_ok and healthy_now))
        action, ec, note = self.decide(o, flags, p_before_lower)
        a, act_note = self.act(action, o)
        note = "; ".join(s for s in [note, act_note] if s)
        rec = {"cycle": len(self.log), "action": action, "note": note, "sensor_ok": sensor_ok,
               **{f"flag_{k}": v for k, v in flags.items()}, **o, **{f"cost_{k}": v for k, v in ec.items()}, **a}
        if truth is not None:
            rec.update(truth)
        self.log.append(rec)
        return rec

    def step_from_features(self, row, truth=None, use_forecast=False, probs=None):
        """Run one OODA pass on a pre-computed per-cycle feature Series/dict without raw samples."""
        flags = self.health(row)
        sensor_ok = not any(flags.values())
        o, p_before_lower, healthy_now = self.orient_from_features(row, use_forecast=use_forecast, probs=probs)
        self.health.update(row, healthy=(sensor_ok and healthy_now))
        action, ec, note = self.decide(o, flags, p_before_lower, allow_plan=use_forecast)
        a, act_note = self.act(action, o)
        note = "; ".join(s for s in [note, act_note] if s)
        rec = {"cycle": len(self.log), "action": action, "note": note, "sensor_ok": sensor_ok,
               **{f"flag_{k}": v for k, v in flags.items()}, **o, **{f"cost_{k}": v for k, v in ec.items()}, **a}
        if truth is not None:
            rec.update(truth)
        self.log.append(rec)
        return rec

    def frame(self):
        return pd.DataFrame(self.log)
