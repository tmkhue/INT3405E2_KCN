"""Unit conversion between normalized vectors and physical units.

What is verified (building_train_1 / fairbanks_1 / pid_water, 8640 steps, and
sanity-checked on building_train_2 / atlanta_3, 600 steps):

* Observation: only some fields are stored normalized. The rules are in
  NORMALIZED_OBS_RULES. Everything else is already in physical units (K, Pa, ...).
  Do NOT infer this from observation_normalization_metadata: that dict lists
  13 groups, but only the ones in NORMALIZED_OBS_RULES are applied to the vector.
* Action: every continuous entry is in [-1, 1] and maps linearly to the native
  range given by action_normalization_metadata.
* Timing: the action sent at step t shows up in the observation of step t+1.
"""
import re

import numpy as np

from src.features import BuildingIndex

# regex on the observation field name -> key in meta["observation_normalization_metadata"]
NORMALIZED_OBS_RULES = [
    (r"^VAV_\d+_zone_air_temperature_sensor$", "zone_air_temperature_sensor"),
    (r"^VAV_\d+_supply_air_damper_percentage_command$", "supply_air_damper_percentage_command"),
    (r"^VAV_\d+_reheat_valve_percentage_command$", "reheat_valve_percentage_command"),
    (r"^hws_differential_pressure_setpoint$", "differential_pressure_setpoint"),
    (r"^hws_run_status$", "run_status"),
    (r"^hws_supervisor_run_command$", "supervisor_run_command"),
]

# The simulator floors every VAV damper at 10% open (normalized -0.8).
DAMPER_FLOOR_NATIVE = 0.10

# A running AHU holds a minimum static pressure. Observed 200 Pa on building_train_1:
# obs setpoint = max(commanded, 200) while the AHU is on, 0 while it is off.
# NOT yet checked on the other buildings.
AHU_MIN_STATIC_PRESSURE_PA = 200.0


class ObsUnits:
    """Convert observation vectors between normalized and native units."""

    def __init__(self, meta):
        names = list(meta["observation_field_names"])
        norm = meta["observation_normalization_metadata"]
        self.n_obs = len(names)
        idx, mean, std = [], [], []
        for i, name in enumerate(names):
            for pattern, key in NORMALIZED_OBS_RULES:
                if re.match(pattern, name):
                    idx.append(i)
                    mean.append(norm[key]["sample_mean"])
                    std.append(norm[key]["sample_std"])
                    break
        self.idx = np.array(idx, dtype=int)
        self.mean = np.array(mean, dtype=np.float64)
        self.std = np.array(std, dtype=np.float64)

    def to_native(self, obs):
        """Return a float64 copy of obs (last axis = fields) in native units."""
        out = np.array(obs, dtype=np.float64, copy=True)
        out[..., self.idx] = out[..., self.idx] * self.std + self.mean
        return out

    def to_normalized(self, native):
        """Inverse of to_native."""
        out = np.array(native, dtype=np.float64, copy=True)
        out[..., self.idx] = (out[..., self.idx] - self.mean) / self.std
        return out


class ActionLayout:
    """Index arrays and native ranges of the continuous action vector."""

    def __init__(self, meta):
        self.index = BuildingIndex(meta)
        self.n_cont = int(meta["n_continuous_action"])
        self.n_disc = int(meta["n_discrete_action"])
        K = self.index.num_ahus
        self.num_ahus = K

        self.ahu_temp = np.array(meta["temp_indices"], dtype=int)
        pressure = list(meta["pressure_indices"])
        if len(pressure) != K + 1:
            raise ValueError(f"expected K+1={K + 1} pressure indices, got {len(pressure)}")
        # first K = AHU static pressure, last = hot-water differential pressure
        self.ahu_pressure = np.array(pressure[:K], dtype=int)
        self.hws_dp = int(pressure[K])
        self.ahu_recirc = np.array(meta["recirculation_indices"], dtype=int)
        if len(meta["water_indices"]) != 1:
            raise ValueError("expected exactly one water index (supply water temperature)")
        self.hws_temp = int(meta["water_indices"][0])
        self.zone_damper = self.index.zone_damper_action  # ordered by zone id
        self.zone_reheat = self.index.zone_reheat_action

        # native ranges
        a = meta["action_normalization_metadata"]
        lo = np.full(self.n_cont, np.nan)
        hi = np.full(self.n_cont, np.nan)
        nlo = np.full(self.n_cont, np.nan)
        nhi = np.full(self.n_cont, np.nan)

        def fill(indices, key):
            m = a[key]
            for i in np.atleast_1d(indices):
                lo[i], hi[i] = m["min_native_value"], m["max_native_value"]
                nlo[i], nhi[i] = m["min_normalized_value"], m["max_normalized_value"]

        # AHU k (0-based position) uses the metadata keys ahu_{k+1}_...
        for k in range(K):
            fill(self.ahu_temp[k], f"ahu_{k + 1}_supply_air_temperature_setpoint")
            fill(self.ahu_pressure[k], f"ahu_{k + 1}_supply_air_static_pressure_setpoint")
            fill(self.ahu_recirc[k], f"ahu_{k + 1}_recirculation_percentage_command")
        fill(self.hws_dp, "differential_pressure_setpoint")
        fill(self.hws_temp, "supply_water_temperature_setpoint")
        fill(self.zone_damper, "supply_air_damper_percentage_command")
        fill(self.zone_reheat, "reheat_valve_percentage_command")

        if np.isnan(lo).any():
            missing = np.where(np.isnan(lo))[0].tolist()
            raise ValueError(f"action entries without a native range: {missing}")
        self.native_lo, self.native_hi = lo, hi
        self.norm_lo, self.norm_hi = nlo, nhi

    def neutral(self):
        """All-zero continuous action: the MIDPOINT of every range, not 'off'."""
        return np.zeros(self.n_cont, dtype=np.float32)

    def to_native(self, cont):
        cont = np.asarray(cont, dtype=np.float64)
        frac = (cont - self.norm_lo) / (self.norm_hi - self.norm_lo)
        return self.native_lo + frac * (self.native_hi - self.native_lo)

    def to_normalized(self, native, clip=True):
        native = np.asarray(native, dtype=np.float64)
        frac = (native - self.native_lo) / (self.native_hi - self.native_lo)
        out = self.norm_lo + frac * (self.norm_hi - self.norm_lo)
        if clip:
            out = np.clip(out, self.norm_lo, self.norm_hi)
        return out.astype(np.float32)


def clock_from_obs(obs, names):
    """Hour of day in [0, 24) and day-of-week index (Monday = 0) from one obs.

    hod_cos_000 / hod_sin_000 encode the CURRENT hour (angle = hour * 15 deg).
    dow_cos_000 / dow_sin_000 encode the current day index (angle = day * 360/7).
    Entries _001 .. _011 are look-ahead copies (hod: +2 h each; dow: +14 h each).
    """
    pos = {n: i for i, n in enumerate(names)}
    ang_h = np.degrees(np.arctan2(obs[pos["hod_sin_000"]], obs[pos["hod_cos_000"]])) % 360.0
    ang_d = np.degrees(np.arctan2(obs[pos["dow_sin_000"]], obs[pos["dow_cos_000"]])) % 360.0
    hour = ang_h / 15.0
    day = int(round(ang_d / (360.0 / 7.0))) % 7
    return float(hour), day
