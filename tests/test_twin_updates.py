"""Unit tests for the Deliverable 3 updates in twin/."""
import unittest
import numpy as np
import pandas as pd

from twin.detect import HealthCheck, FEATURES, make_classifier
from twin.forecast import RestartingTrendKF
from twin.decide import DecisionRule, realised_cycle_cost
from twin.loop import OODATwin


class TwinUpdatesTest(unittest.TestCase):
    def test_adaptive_health_check_tightens_within_batch_band(self):
        rng = np.random.default_rng(0)
        rows = []
        for variant, base_ab in (("PRE_A", 0.50), ("PRE_B", 0.36)):
            for _ in range(80):
                rows.append({
                    "variant": variant,
                    "sensor_ab_mean": base_ab + rng.normal(0, 0.055),
                    "sensor_ab_std": 0.65 + rng.normal(0, 0.02),
                    "duration": 11.0 + rng.normal(0, 0.15),
                })
        ref = pd.DataFrame(rows)
        hc_static = HealthCheck(ref, k=4.0, adaptive=False)
        hc_adapt = HealthCheck(ref, k=4.0, adaptive=True)
        self.assertLess(hc_adapt.band["sensor_ab_mean"][1], hc_static.band["sensor_ab_mean"][1])
        # A 0.5 N m offset is flagged cleanly by adaptive check
        faulty = {"sensor_ab_mean": hc_adapt.band["sensor_ab_mean"][0] + 0.5, "sensor_ab_std": 0.65, "duration": 11.0}
        self.assertTrue(hc_adapt(faulty)["sensor_ab_mean"])

    def test_slope_gating_waits_for_confirmed_drift(self):
        rng = np.random.default_rng(1)
        kf = RestartingTrendKF(level0=0.58, ref_mean=0.58, ref_std=0.17, k_onset=1.5, n_onset=3,
                               obs_std=0.17, q_level=0.01, q_slope=1e-4)
        for _ in range(40):
            kf.step(0.58 + rng.normal(0, 0.17))
        self.assertFalse(kf.is_ready())
        for k in range(1, 90):
            kf.step(0.58 + 0.012 * k + rng.normal(0, 0.17))
        self.assertTrue(kf.is_ready(min_cycles=10, p_slope_min=0.90))
        self.assertGreater(kf.p_positive_slope(), 0.90)

    def test_severe_override_skips_hysteresis_and_realised_cost(self):
        rule = DecisionRule(hysteresis=2)
        a_norm, _, _ = rule.decide(0.95, 0.95, 0.0, severe=False)
        self.assertEqual(a_norm, "slow")
        rule2 = DecisionRule(hysteresis=2)
        a_sev, _, note = rule2.decide(0.95, 0.95, 0.0, severe=True)
        self.assertEqual(a_sev, "stop")
        self.assertIn("override", note)
        self.assertEqual(realised_cycle_cost("stop", "friction"), 3.0)
        self.assertEqual(realised_cycle_cost("stop", "healthy"), 30.0)
        self.assertEqual(realised_cycle_cost("continue", "friction"), 100.0)

    def test_step_from_features_uses_relative_limit(self):
        rng = np.random.default_rng(2)
        n = 60
        df = pd.DataFrame({c: rng.normal(0, 0.1, n) for c in FEATURES})
        df["load_coulomb"] = 0.58 + rng.normal(0, 0.15, n)
        df["load_c0"] = -6.9 + rng.normal(0, 0.08, n)
        df["sensor_ab_mean"] = 0.48 + rng.normal(0, 0.05, n)
        df["sensor_ab_std"] = 0.65 + rng.normal(0, 0.02, n)
        df["duration"] = 11.0 + rng.normal(0, 0.1, n)
        df["variant"] = "PRE_A"
        y = np.array(["healthy"] * 45 + ["friction"] * 15, dtype=object)
        df.loc[45:, "load_coulomb"] = 2.0 + rng.normal(0, 0.15, 15)
        clf = make_classifier("nb", cv=3).fit(df[FEATURES], y)
        tw = OODATwin(df.iloc[:45], clf, tau_max=1.8, delta_tau_max=1.0, severe_z=6.0, adaptive_health=True, gate_forecast=True)
        rec_h = tw.step_from_features(df.iloc[0])
        self.assertEqual(rec_h["action"], "continue")
        rec_f = tw.step_from_features(df.iloc[50])
        self.assertEqual(rec_f["action"], "stop")
        self.assertTrue(rec_f["severe_override"])


if __name__ == "__main__":
    unittest.main()
