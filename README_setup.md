# Smart Buildings – Setup Guide

## 1. Environment

- Operating System: Windows 10
- Python: 3.12.10
- Virtual environment: `.venv`
- Python environment creation:

```powershell
py -3.12 -m venv .venv
```

## 2. Repository Information

- Repository: https://github.com/tmkhue/INT3405E2_KCN
- Starter-kit commit SHA: `1fb75606d12e016463fbc63de704967c5b2bf9d5`

## 3. Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate the environment again.

## 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -e .
pip freeze > requirements_frozen.txt
```

## 5. Run a Simulation Scenario

```powershell
python -m smart_buildings.run_environment --building building_train_1 --weather fairbanks_1
```

Save the output:

```powershell
New-Item -ItemType Directory -Force logs
python -m smart_buildings.run_environment --building building_train_1 --weather fairbanks_1 *> logs/run_env_b1.txt
```

## 6. Evaluate the Default Policy

Run a one-day smoke test (288 steps):

```powershell
python -m smart_buildings.evaluate --policies transfer --buildings building_train_1 --weathers fairbanks_1 --steps 288
```

Results are saved under:

`smart_buildings/results/<building>/<weather>/<policy>/summary.json`

**Warning:** The reward obtained from a 288-step smoke test is only used to verify that the simulator and policy run successfully. It must not be used to compare policy performance.

## 7. Runtime Notes

Approximate runtime observations:

- Scenario loading: approximately 50 seconds.
- A 10-step simulation: approximately 5 seconds.

Actual runtime may vary depending on hardware and system load.

## 8. Project Files

- `requirements_frozen.txt`: records installed dependency versions.
- `logs/`: stores simulator execution logs.
- `.venv/`: local Python virtual environment.

These local environment and log files should not be committed to Git.
