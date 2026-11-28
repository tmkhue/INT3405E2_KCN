"""Index tables for the observation and action vectors, built from building_meta_data.

Never hardcode lengths or offsets: every index comes from the field names and metadata.
"""
import re
from collections import defaultdict

import numpy as np

_VAV_RE = re.compile(r"^VAV_(\d+)_(.+)$")
_AHU_RE = re.compile(r"^ahs_ahu_(\d+)_(.+)$")
_ROOM_RE = re.compile(r"^room_(\d+)$")
_AHU_KEY_RE = re.compile(r"^ahu_(\d+)$")

ZONE_FIELDS = {
    "reheat": "reheat_valve_percentage_command",
    "damper": "supply_air_damper_percentage_command",
    "flow_sp": "supply_air_flowrate_setpoint",
    "cool_sp": "zone_air_cooling_temperature_setpoint_sensor",
    "heat_sp": "zone_air_heating_temperature_setpoint_sensor",
    "temp": "zone_air_temperature_sensor",
}


class BuildingIndex:
    """Observation/action indices of one building. Zones are ordered numerically (1, 2, 3, ...)."""

    def __init__(self, meta):
        self.meta = meta
        names = list(meta["observation_field_names"])
        self.n_obs = len(names)

        vav = defaultdict(dict)    # zone_id -> {field suffix: obs index}
        ahu = defaultdict(dict)    # ahu_id  -> {field suffix: obs index}
        self.other = {}            # remaining fields (hws_..., hod_..., ...)
        for i, name in enumerate(names):
            m = _VAV_RE.match(name)
            if m:
                vav[int(m.group(1))][m.group(2)] = i
                continue
            m = _AHU_RE.match(name)
            if m:
                ahu[int(m.group(1))][m.group(2)] = i
                continue
            self.other[name] = i

        self.zone_ids = sorted(vav)
        self.ahu_ids = sorted(ahu)
        self._vav, self._ahu = vav, ahu

        # zone -> AHU (room_N <-> VAV_N)
        self.ahu_of_zone = {}
        self.zones_of_ahu = {}
        for ahu_key, rooms in meta["ahu_zone_assignments"].items():
            a = int(_AHU_KEY_RE.match(ahu_key).group(1))
            zs = [int(_ROOM_RE.match(r).group(1)) for r in rooms]
            self.zones_of_ahu[a] = sorted(zs)
            for z in zs:
                self.ahu_of_zone[z] = a

        # obs index arrays, ordered by zone_ids
        self.zone_obs = {
            key: np.array([vav[z][suffix] for z in self.zone_ids])
            for key, suffix in ZONE_FIELDS.items()
        }

        # action index arrays, ordered by zone_ids
        zai = meta["zone_action_indices"]
        self.zone_damper_action = np.array(
            [zai[f"VAV_{z}"]["supply_air_damper_percentage_command"] for z in self.zone_ids])
        self.zone_reheat_action = np.array(
            [zai[f"VAV_{z}"]["reheat_valve_percentage_command"] for z in self.zone_ids])

    @property
    def num_zones(self):
        return len(self.zone_ids)

    @property
    def num_ahus(self):
        return len(self.ahu_ids)

    def ahu_obs(self, suffix):
        """Obs indices of one AHU field, ordered by ahu_ids."""
        return np.array([self._ahu[a][suffix] for a in self.ahu_ids])

    def all_action_indices(self):
        m = self.meta
        idx = list(m["temp_indices"]) + list(m["pressure_indices"]) \
            + list(m["recirculation_indices"]) + list(m["water_indices"]) \
            + list(self.zone_damper_action) + list(self.zone_reheat_action)
        return sorted(int(i) for i in idx)


def _norm(meta, key):
    s = meta["observation_normalization_metadata"][key]
    return s["sample_mean"], s["sample_std"]


def zone_temp_kelvin(obs, idx):
    """Zone temperatures in Kelvin. obs may have leading batch dimensions."""
    mean, std = _norm(idx.meta, "zone_air_temperature_sensor")
    return obs[..., idx.zone_obs["temp"]] * std + mean


def zone_damper_native(obs, idx):
    """Damper position in 0..1 (obs is normalized to [-1, 1])."""
    mean, std = _norm(idx.meta, "supply_air_damper_percentage_command")
    return obs[..., idx.zone_obs["damper"]] * std + mean


def zone_reheat_native(obs, idx):
    """Reheat valve position in 0..1 (obs is normalized to [-1, 1])."""
    mean, std = _norm(idx.meta, "reheat_valve_percentage_command")
    return obs[..., idx.zone_obs["reheat"]] * std + mean


def hws_dp_setpoint_native(obs, idx):
    mean, std = _norm(idx.meta, "differential_pressure_setpoint")
    return obs[..., idx.other["hws_differential_pressure_setpoint"]] * std + mean


def zone_band_kelvin(obs, idx):
    """(heating_setpoint, cooling_setpoint) per zone. Already raw Kelvin."""
    return obs[..., idx.zone_obs["heat_sp"]], obs[..., idx.zone_obs["cool_sp"]]