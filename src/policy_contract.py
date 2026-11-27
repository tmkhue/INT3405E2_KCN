"""Checks that a policy obeys the interface the evaluator and Kaggle expect.

Contract (read from smart_buildings/evaluate.py):
  * constructed once per scenario with (building_meta_data, action_spec),
    or (building_meta_data), or ();
  * get_action(obs) is called once per step with the observation vector;
  * it must return {"continuous_action": float32 (3K+2+2V,),
                    "discrete_action":   int32   (K+1,)}.
validate_action() in evaluate.py only casts dtypes and checks shapes. It does NOT
clip to [-1, 1], and the discrete part must be exactly 0 or 1.
"""
import numpy as np


class PolicyContractError(ValueError):
    pass


def check_action(action, meta, tol=1e-6, strict_dtype=True):
    """Check one action. Raises PolicyContractError on any problem.

    strict_dtype=False casts first (evaluate.validate_action does that cast), so
    a policy that returns float64/int64 passes. Use strict_dtype=True for your own
    final policy: it is the safe choice for Kaggle, whose wrapper is not verified.
    """
    if not isinstance(action, dict):
        raise PolicyContractError(f"action must be a dict, got {type(action).__name__}")
    for key in ("continuous_action", "discrete_action"):
        if key not in action:
            raise PolicyContractError(f"missing key {key}")
    cont = np.asarray(action["continuous_action"])
    disc = np.asarray(action["discrete_action"])
    if not strict_dtype:
        cont, disc = cont.astype(np.float32), disc.astype(np.int32)
    if cont.dtype != np.float32:
        raise PolicyContractError(f"continuous_action dtype {cont.dtype}, expected float32")
    if disc.dtype != np.int32:
        raise PolicyContractError(f"discrete_action dtype {disc.dtype}, expected int32")
    if cont.shape != (int(meta["n_continuous_action"]),):
        raise PolicyContractError(f"continuous_action shape {cont.shape}, expected ({meta['n_continuous_action']},)")
    if disc.shape != (int(meta["n_discrete_action"]),):
        raise PolicyContractError(f"discrete_action shape {disc.shape}, expected ({meta['n_discrete_action']},)")
    if not np.isfinite(cont).all():
        raise PolicyContractError("continuous_action contains NaN or inf")
    if cont.min() < -1.0 - tol or cont.max() > 1.0 + tol:
        raise PolicyContractError(f"continuous_action outside [-1, 1]: min={cont.min()}, max={cont.max()}")
    if not np.isin(disc, (0, 1)).all():
        raise PolicyContractError(f"discrete_action must be 0/1, got {np.unique(disc).tolist()}")


def check_policy(make_policy, meta, observations, strict_dtype=True):
    """Run a policy over recorded observations and check the contract.

    make_policy: zero-argument function returning a fresh policy instance.
    observations: 2-D array, one observation per row.
    Checks: valid action every step; obs not modified; a fresh instance
    reproduces the first actions (no hidden state leaking between episodes).
    """
    obs_arr = np.asarray(observations, dtype=np.float32)
    first = make_policy()
    produced = []
    for t in range(len(obs_arr)):
        obs = obs_arr[t].copy()
        before = obs.copy()
        action = first.get_action(obs)
        check_action(action, meta, strict_dtype=strict_dtype)
        if not np.array_equal(obs, before):
            raise PolicyContractError(f"policy modified the observation at step {t}")
        produced.append(
            (np.array(action["continuous_action"]), np.array(action["discrete_action"]))
        )
    second = make_policy()
    for t in range(min(len(obs_arr), 5)):
        action = second.get_action(obs_arr[t].copy())
        if not (np.allclose(action["continuous_action"], produced[t][0])
                and np.array_equal(action["discrete_action"], produced[t][1])):
            raise PolicyContractError(
                f"fresh instance gives a different action at step {t}: state leaks or is not reset"
            )
    return True
