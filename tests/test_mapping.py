"""Test mapping trên ít nhất 2 tòa nhà. Chạy: python -m pytest tests -v -s
(mỗi tòa nạp mất khoảng 1 phút)"""
from pathlib import Path

import pytest

from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data
from src.features import BuildingIndex

ROOT = Path("smart_buildings/training_scenarios")
_cache = {}


def get(building):
    if building not in _cache:
        weather = sorted(p.name for p in (ROOT / building).iterdir() if p.is_dir())[0]
        config = ROOT / building / weather / "config.gin"
        env = load_hybrid_environment(config)
        meta = create_building_meta_data(env, config.parent)
        _cache[building] = (meta, BuildingIndex(meta))
    return _cache[building]


@pytest.mark.parametrize("building", [
    "building_train_1", "building_train_2", "building_train_4",
    "building_train_5", "building_train_7", "building_train_8"])
def test_building(building):
    meta, idx = get(building)
    K, V = idx.num_ahus, idx.num_zones

    # kích thước khớp công thức
    assert meta["n_continuous_action"] == 3 * K + 2 + 2 * V
    assert meta["n_discrete_action"] == K + 1

    # chỉ số action phủ đúng từng ô một lần
    assert idx.all_action_indices() == list(range(meta["n_continuous_action"]))

    # zone_temp_indices của metadata trùng chỉ số tìm từ tên trường
    assert sorted(meta["zone_temp_indices"]) == sorted(idx.zone_obs["temp"].tolist())

    # mọi chỉ số obs nằm trong phạm vi và mỗi trường chỉ một lần
    for arr in idx.zone_obs.values():
        assert len(arr) == V and arr.min() >= 0 and arr.max() < idx.n_obs
    all_zone = sum((a.tolist() for a in idx.zone_obs.values()), [])
    assert len(all_zone) == len(set(all_zone))

    # mọi zone thuộc đúng một AHU, và AHU khớp với metadata
    assert set(idx.ahu_of_zone) == set(idx.zone_ids)
    assert sum(len(v) for v in idx.zones_of_ahu.values()) == V