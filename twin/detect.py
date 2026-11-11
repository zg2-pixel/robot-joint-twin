"""Detection layer: health checks on the sensors, a residual test for 'something changed', and a calibrated
classifier for 'what changed'. Everything works on the per-cycle table of twin/features.py."""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix

CLASSES = ["healthy", "friction", "payload", "collision", "electrical"]

# per-cycle features the classifier may see. Load-side physics first, then motor side, then shape of the cycle.
# No session-level quantities (gear viscous friction, A-B offset): those drift between recording batches and
# are handled by the reference tracker and the health check instead.
FEATURES = ["load_coulomb", "load_c0", "load_viscous", "load_noise", "load_rms", "resid_peak", "resid_p99",
            "gear_coulomb", "iq_rms", "torque_rms_ac", "torque_peak",
            "track_err_rms", "track_err_max", "duration", "moving_frac", "speed_peak"]


def label_for_joint(T, axis):
    """Class label of each cycle from the point of view of one joint."""
    lab = pd.Series("healthy", index=T.index)
    own = T["variant_axis"] == axis
    lab[(T.category_name == "axis_friction") & own] = "friction"
    lab[(T.category_name == "axis_weight") & (T["variant_axis"] != 1)] = "payload"     # mass on axis 1 does not load joint 2+
    lab[T.category_name == "can_weight"] = "payload"
    lab[T.category_name.str.startswith("collision")] = "collision"
    lab[(T.category_name == "motor_commutation") & own] = "electrical"
    unseen = T.category_name.isin(["lose_can", "miss_can", "entangled", "invalid_position", "wobbling_station"])
    lab[unseen] = "unseen"
    # friction / commutation injected on another axis is 'healthy' for this joint
    return lab


class HealthCheck:
    """Are the sensors to be trusted this cycle? Compares the A-B disagreement (mean and spread) and the cycle
    time with their healthy band (mean +- k * spread, from the reference cycles)."""

    def __init__(self, ref, k=4.0):
        self.k = k
        self.band = {c: (ref[c].mean(), ref[c].std()) for c in ["sensor_ab_mean", "sensor_ab_std", "duration"]}

    def __call__(self, row):
        flags = {}
        for c, (m, s) in self.band.items():
            flags[c] = abs(row[c] - m) > self.k * s
        return flags


class ResidualTest:
    """'Something changed' without knowing what: the per-cycle fit residual against its healthy distribution,
    both its overall size (load_noise) and its worst short burst (resid_peak)."""

    def __init__(self, ref, k=4.0):
        self.k = k
        self.band = {c: (ref[c].mean(), ref[c].std()) for c in ["load_noise", "resid_peak"]}

    def z(self, row):
        return max((row[c] - m) / s for c, (m, s) in self.band.items())

    def flag(self, row):
        return self.z(row) > self.k


def make_classifier(kind="nb", k=5, calibrate=True, cv=5):
    """Gaussian naive Bayes or k-NN on standardised features, with Platt-style probability calibration."""
    base = GaussianNB() if kind == "nb" else KNeighborsClassifier(n_neighbors=k, weights="distance")
    pipe = make_pipeline(StandardScaler(), base)
    if calibrate:
        return CalibratedClassifierCV(pipe, method="sigmoid", cv=cv)
    return pipe


def cv_probabilities(X, y, kind="nb", k=5, folds=5, seed=0):
    """Out-of-fold class probabilities, so every cycle is scored by a model that never saw it."""
    y = np.asarray(y, dtype=object)
    clf = make_classifier(kind, k=k)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    P = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    classes = sorted(set(y))
    return pd.DataFrame(P, columns=classes, index=X.index)


def summarize(P, y):
    """Per-class recall, overall accuracy, Brier score of the probability vectors, confusion matrix."""
    y = np.asarray(y, dtype=object)
    pred = P.idxmax(axis=1).to_numpy(dtype=object)
    classes = list(P.columns)
    cm = confusion_matrix(y, pred, labels=classes)
    recall = {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else np.nan for i, c in enumerate(classes)}
    onehot = pd.get_dummies(pd.Series(y, index=P.index)).reindex(columns=classes, fill_value=0).values.astype(float)
    brier = float(np.mean(np.sum((P.values - onehot) ** 2, axis=1)))
    return {"accuracy": float((pred == y).mean()), "recall": recall, "brier": brier,
            "confusion": pd.DataFrame(cm, index=classes, columns=classes)}


def reliability(p, truth, bins=10):
    """Reliability curve of a probability p for a binary truth: mean predicted vs observed frequency per bin."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() >= 5:
            out.append((float(p[m].mean()), float(truth[m].mean()), int(m.sum())))
    return pd.DataFrame(out, columns=["predicted", "observed", "n"])
