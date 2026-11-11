# robot-joint-twin

Digital twin of one joint of a robot arm that detects faults, predicts when wear will cross a limit, and
recommends what to do (continue / slow down / schedule maintenance / stop). Final project for 12-831
Digital Twins and AI for Predictive Analytics, CMU. Team: Zhaoyang Guo, Zhengjie Cui, Junchi Fan.

Data: the [voraus-AD dataset](https://github.com/vorausrobotik/voraus-ad-dataset) (Brockmann et al.,
IEEE T-RO 2024), a six-axis cobot doing pick-and-place: 1367 normal cycles, 755 cycles with 12 kinds of
injected anomalies, 100 Hz, every axis with position, velocity, motor current, motor torque, two joint torque
sensors and the controller's own model torque. See `data/README.md` for the download.

## Idea

For every axis the drive gives three torques: what the controller's rigid-body model expects
(`computed_torque`), what the joint torque sensor measures (`torque_sensor_a/b`) and what the motor
delivers (`motor_torque`). The differences are friction:

    sensor - computed = c0 + tau_c * sign(v) + b * v + dJ * a      (load side)
    motor  - sensor   = gearbox friction, same form                (gear side)

Fitting this on every cycle gives a few physical numbers per cycle. The Coulomb term `tau_c` on the load
side is the health indicator: it jumps when friction is added to that axis and stays put when the fault
is somewhere else. Everything downstream (classifier, forecast, decision) works on these per-cycle numbers.

`tau_c` is an effective number, not the true friction torque: on joint 3 it comes out negative, so the
controller's model torque probably has some friction in it already. We only use it as an indicator, what
counts is that it shifts when friction is added.

## Layout

    twin/dataset.py       load one axis from the parquet, label names, which axis a fault targets
    twin/physics.py       friction fit per cycle, with standard errors
    twin/features.py      per-cycle feature table
    twin/uncertainty.py   quantization, noise at rest, spectra, sampling statistics
    twin/bayes.py         conjugate Bayesian regression with forgetting; the reference tracker
    twin/detect.py        health checks, residual test, NB / k-NN classifier with calibration
    twin/joint_sim.py     joint simulator: real cycles plus the torque of extra friction, for long degradation runs
    twin/forecast.py      Kalman trend on tau_c (restarted when a drift sets in) and Monte Carlo cycles-to-limit
    twin/decide.py        expected-cost rule (continue / slow / stop), lead-time rule for scheduling, hysteresis, abstain
    twin/loop.py          the OODA loop: one pass per cycle, same code for recorded and simulated cycles
    scripts/01_explore.py       step 1: overview + per-cycle fits + figures      ->  results/01_explore/
    scripts/02_uncertainty.py   step 2: resolution, noise, bias, drift, spectra  ->  results/02_uncertainty/
    scripts/03_detect.py        step 3: reference tracking, detection, classifier ->  results/03_detect/
    scripts/04_forecast.py      step 4: simulator, degradation runs, forecast coverage -> results/04_forecast/
    scripts/05_replay.py        step 5: OODA replay, scenarios S1-S6, baselines B1/B2  -> results/05_replay/
    dashboard/app.py            step 6: the dashboard (streamlit), replays results/05_replay/replay_log.csv
    dashboard/twin_view.py      figures for the dashboard and the schematic 3D arm (plotly)
    results/              outputs of the scripts (csv, json, png)

## Run

    pip install -r requirements.txt
    python scripts/01_explore.py            # ~10 s, all six axes
    python scripts/02_uncertainty.py        # ~30 s, all six axes
    python scripts/03_detect.py             # ~10 s, joint 2
    python scripts/04_forecast.py           # ~30 s, 50 simulated degradation runs
    python scripts/05_replay.py             # ~1-2 min, the whole loop on six scenarios
    streamlit run dashboard/app.py          # opens in the browser; needs the results of step 5

## Dashboard

One screen per cycle: the arm in 3D with joint 2 coloured by its health, the friction history with the
healthy band and the limit, the forecast of cycles-to-limit with its 90 % interval, what the classifier
thinks the cycle is, the expected cost of each action, and the action band over the whole scenario with the
points where the two baselines would have stopped. Pick a scenario (recorded faults, simulated wear runs,
sensor drift, an unseen fault), step through the cycles or let it play, change the cost ratios to see how
the decisions move, acknowledge a maintenance ticket, download the log.

The arm uses the recorded joint angles of each cycle (the first start reads them from the parquet once and
caches them in `results/06_dashboard/poses.parquet`, which is not in git); without the data file it falls
back to a synthetic pick-and-place. Link lengths and angle offsets are schematic, not the real kinematics.

## Status

- [x] step 1: dataset overview, friction fit per cycle, health indicator check
- [x] step 2: signal resolution and noise floors, sensor biases, batch-to-batch drift, spectra, propagation into the friction estimate (the uncertainty budget)
- [x] step 3: Bayesian tracking of the healthy state through the recording batches, change detection on the injected faults, calibrated NB / k-NN classifier, residual test on unseen anomalies
- [x] step 4: joint simulator checked against the healthy cycles, 50 simulated degradation runs, Kalman trend + Monte Carlo cycles-to-limit with 90 % intervals, coverage check
- [x] step 5: decision layer and the whole OODA loop replayed on six scenarios against a threshold baseline and a classifier-only baseline
- [x] step 6: dashboard with a 3D view of the arm, replaying the step-5 log cycle by cycle
- [ ] step 7: validation and maintenance plan for the twin itself (deliverable 4)

## Data and licence

The voraus-AD dataset is by voraus robotik and Leibniz University Hannover (Brockmann et al., IEEE T-RO 2024)
and is licensed CC BY-NC-SA 4.0. The raw file is not in this repository, see `data/README.md`. The tables
and figures in `results/` are computed from it and are shared under the same licence, non-commercial use only.
