"""Decision layer: turn the predictions of one cycle into one of four actions by expected cost.

Costs are in units of 'one cycle of production' (to be agreed with the plant; the ratios are what matters):
  missed    a failure while the line keeps running: ~100 cycles of line stop plus the repair
  stop      an unplanned stop that turns out to be unnecessary: ~30 cycles (stop, inspect, restart)
  sched     a planned maintenance inside a predicted window: ~3 cycles
  slow      one cycle at 70 % speed: 0.3 of a cycle lost
  waste     one cycle of useful life thrown away by stopping or servicing earlier than needed: 0.1
Two kinds of decision live here.
The acute one (continue / slow / stop) is by expected cost over the lead time L the plant needs to arrange a
planned stop:  p_fail = P(fault now) + (1 - P(fault now)) * P(N* <= L). With the numbers above the rule climbs a
ladder as the risk grows: continue below ~12 %, slow between ~12 and ~30 %, stop above.
The planning one (schedule) is a lead-time rule on the forecast: a ticket is issued once the forecast says there
is at least p_commit (15 %) probability of crossing the limit within L + plan_margin cycles, and the maintenance
is placed at the lower bound of the 90 % interval of N* (never earlier than L). Waiting until the risk is real
keeps the useful life; placing the stop at the lower bound keeps the chance of failing first at ~5 %. If the lower
bound is already inside L the stop can only be an unplanned one, which is what the acute rule then produces.
A stop needs the same verdict on two consecutive cycles (hysteresis); a failed sensor check means abstain.
"""
import numpy as np

ACTIONS = ["continue", "slow", "schedule", "stop"]
DEFAULT_COSTS = {"missed": 100.0, "stop": 30.0, "sched": 3.0, "slow": 0.3, "waste": 0.1}


class DecisionRule:
    def __init__(self, costs=None, lead=30, plan_margin=20, p_commit=0.15, slow_relief=0.5, hysteresis=2):
        self.c = dict(DEFAULT_COSTS, **(costs or {}))
        self.W = lead                            # the acute window is the lead time
        self.plan_margin = plan_margin
        self.p_commit = p_commit
        self.slow_relief = slow_relief          # slowing down is assumed to halve the risk inside the window
        self.hysteresis = hysteresis
        self.stop_votes = 0
        self.last = "continue"

    def expected_costs(self, p_fault, p_cross_window, p_cross_plan, unused_life=0.0):
        """Expected cost of each action. p_cross_window: P(N* <= W); p_cross_plan: P(N* <= planned date);
        unused_life: expected cycles of life given up by servicing at the planned date."""
        p_fail = p_fault + (1 - p_fault) * p_cross_window
        return {
            "continue": p_fail * self.c["missed"],
            "slow": self.slow_relief * p_fail * self.c["missed"] + self.W * self.c["slow"],
            "schedule": self.c["sched"] + (p_fault + (1 - p_fault) * p_cross_plan) * self.c["missed"] + unused_life * self.c["waste"],
            "stop": (1 - p_fault) * self.c["stop"] + p_fault * self.c["sched"],
        }

    def plan_now(self, p_cross_plan_horizon, ticket_open):
        """The lead-time rule: issue the ticket once crossing within L + margin has become likely enough."""
        return (not ticket_open) and (p_cross_plan_horizon >= self.p_commit)

    def decide(self, p_fault, p_cross_window, p_cross_plan, sensor_ok=True, unused_life=0.0, lo5=np.inf, ticket_open=False, p_cross_plan_horizon=0.0):
        """Returns (action, expected costs, note)."""
        if not sensor_ok:
            self.stop_votes = 0
            return "abstain", {}, "sensor check failed: no decision, request a sensor inspection"
        ec = self.expected_costs(p_fault, p_cross_window, p_cross_plan, unused_life)
        acute = {a: ec[a] for a in ["continue", "slow", "stop"]}
        best = min(acute, key=acute.get)
        note = ""
        if best != "stop" and self.plan_now(p_cross_plan_horizon, ticket_open):
            self.stop_votes = 0
            self.last = "schedule"
            return "schedule", ec, f"P(cross within {self.W + self.plan_margin}) = {p_cross_plan_horizon:.2f}: plan at {max(lo5, self.W):.0f} cycles (lower bound of the 90 % interval)"
        if best == "stop":
            self.stop_votes += 1
            if self.stop_votes < self.hysteresis:
                # not yet: take the next best action this cycle
                best = min((a for a in acute if a != "stop"), key=acute.get)
                note = f"stop needs {self.hysteresis} consecutive votes ({self.stop_votes} so far)"
        else:
            self.stop_votes = 0
        self.last = best
        return best, ec, note
