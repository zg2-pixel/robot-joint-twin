"""A simulator of one joint, built on the same equation the twin fits.

The idea is the one the dataset authors used with a foam pad, done in software: take a real healthy cycle and add
the torque that extra friction would cost, tau_c * sign(v) (plus any change of c0 or b), to the measured torques.
Everything else - trajectory, controller model, the repeatable model error, the sensor noise - stays exactly as
recorded, so healthy synthetic cycles are real cycles and degraded ones differ from them only by physics we
control. That gives long runs with a *known* slow drift of tau_c, which the dataset does not contain.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

MOVING = 0.02


class JointSimulator:
    def __init__(self, templates, healthy, seed=0):
        """templates: list of real healthy cycle frames. healthy: (c0, tau_c, b) of the healthy joint, the
        values the increments are measured from."""
        self.templates = templates
        self.c0, self.tau_c, self.b = healthy
        self.rng = np.random.default_rng(seed)

    def cycle(self, tau_c=None, c0=None, b=None, sensor_offset=0.0, speed=1.0, template=None, extra_noise_std=0.0, rho=0.86):
        """One synthetic cycle as a frame with the columns the twin reads.
        tau_c / c0 / b: the joint's parameters for this cycle (default: healthy, i.e. the real cycle untouched).
        sensor_offset: added to torque sensor A only (a sensor fault). speed < 1: the motion is slowed down
        (time stretched, velocities and accelerations scaled; the friction torques follow the new velocity).
        extra_noise_std: additional coloured noise on the load torque (a worse sensor / rougher process)."""
        g = self.templates[self.rng.integers(len(self.templates))] if template is None else template
        d_tc = (self.tau_c if tau_c is None else tau_c) - self.tau_c
        d_c0 = (self.c0 if c0 is None else c0) - self.c0
        d_b = (self.b if b is None else b) - self.b
        v0 = g["joint_velocity"].values
        v = v0 * speed
        moving = np.abs(v) > MOVING
        extra = d_c0 + d_tc * np.sign(v) * moving + d_b * v
        if speed != 1.0:
            # the healthy friction also follows the slower velocity: scale its viscous part, keep the Coulomb part
            extra = extra + self.b * (v - v0)
        if extra_noise_std > 0:
            w = self.rng.normal(0, extra_noise_std * np.sqrt(1 - rho ** 2), len(v) + 50)
            extra = extra + lfilter([1.0], [1.0, -rho], w)[50:]
        out = pd.DataFrame({
            "time": g["time"].values / speed,
            "joint_velocity": v,
            "target_velocity": g["target_velocity"].values * speed,
            "target_acceleration": g["target_acceleration"].values * speed ** 2,
            "computed_torque": g["computed_torque"].values,
            "torque_sensor_a": g["torque_sensor_a"].values + extra + sensor_offset,
            "torque_sensor_b": g["torque_sensor_b"].values + extra,
            "motor_torque": g["motor_torque"].values + extra,
            "motor_iq": g["motor_iq"].values + extra / 14.3,          # joint 2: about 14.3 N m of motor torque per A
            "joint_position": g["joint_position"].values,
            "target_position": g["target_position"].values,
            "robot_current": g["robot_current"].values,
            "active": g["active"].values,
        })
        return out

    def degradation_run(self, n_cycles, rate, walk_std=0.0, tau_start=None, seed=None, extra_noise_std=0.0):
        """A sequence of cycles whose tau_c drifts: tau_c(k) = tau_start + rate * k + random walk.
        Returns the frames and the true tau_c of every cycle."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        tau = self.tau_c if tau_start is None else tau_start
        truth, frames = [], []
        for _ in range(n_cycles):
            truth.append(tau)
            frames.append(self.cycle(tau_c=tau, extra_noise_std=extra_noise_std))
            tau = tau + rate + self.rng.normal(0, walk_std)
        return frames, np.array(truth)

    def degradation_path(self, n_cycles, rate, walk_std=0.0, tau_start=None, seed=None):
        """Only the true tau_c path (the frames are then generated one by one, so the Act step can slow them down)."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        tau = self.tau_c if tau_start is None else tau_start
        path = []
        for _ in range(n_cycles):
            path.append(tau)
            tau = tau + rate + rng.normal(0, walk_std)
        return np.array(path)
