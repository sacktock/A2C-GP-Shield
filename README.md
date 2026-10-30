# GP-Shield

Safe reinforcement learning via recovery-based shielding with Gaussian Process dynamics models.

This repository is the training code for the paper "Safe Reinforcement Learning via Recovery-based Shielding with Gaussian Process Dynamics Models". Environments use the [Gymnasium](https://gymnasium.farama.org/) interface. The shield is a Gaussian-process dynamics model that overrides the learned policy with a safe backup controller.

## Quick Start

### Installation

#### Prerequisites

Python 3.12 is required.

#### Installation with conda

- Install conda, for example via [Miniconda](https://docs.anaconda.com/miniconda/).
- Clone the repo and create the environment from the repository root:

```bash
conda env create --name gpshield --file conda-environment.yaml
conda activate gpshield
```

- Install the project and its dependencies:

```bash
pip install -e .
```

`pip install -e .` installs the dependencies declared in `pyproject.toml`. `requirements.txt` lists the same base dependencies and is kept in lockstep with that file.

#### Installation with uv

From the repository root, `uv sync` installs the base dependency set (CPU JAX):

```bash
uv sync
```

Install the test extra as well:

```bash
uv sync --group dev
```

GPU JAX is a separate dependency group. Pick the one that matches your driver:

```bash
uv sync --group cuda12
```

```bash
uv sync --group cuda13
```

`cuda12` and `cuda13` conflict with each other. Do not enable both.

#### Enabling GPU acceleration with JAX (optional)

GP-Shield uses [JAX](https://docs.jax.dev/) for the dynamics model and the actor-critic. CPU is enough to run the environments and a short training job.

**macOS and Windows:** use the CPU install above. No extra JAX step is required.

**Linux x86_64 / aarch64 and Windows WSL2:** install the CUDA extra that matches your driver. These groups follow the JAX version in the lock, so they stay compatible with `gpjax==1.0.0` (`jax<0.11`).

```bash
uv sync --group cuda12
```

```bash
uv sync --group cuda13
```

Direct dependencies have lower bounds. `gpjax` stays pinned at `1.0.0`, which still requires `jax<0.11`, so the lock resolves `jax` and `jaxlib` to 0.10.2 (the newest release inside that bound, and the one that ships both CUDA extras). `numpyro` is at least 0.21 so it imports against that JAX. `tensorflow` resolves to 2.21.0.

MuJoCo (used by Hopper) is installed through the `gymnasium[mujoco]` extra. On a headless Linux machine, also install OSMesa and select it, or `Hopper-v5` fails while importing its renderer:

```bash
sudo apt-get install -y libosmesa6 libgl1 libglib2.0-0
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

## How to run GP-Shield

Training is configured from `gpshield/configs.yaml`. The console script is available after installation:

```bash
gpshield --configs cartpole
```

With uv, without activating another environment:

```bash
uv run gpshield --configs cartpole
```

Any key in the loaded config can be overridden from the command line:

```bash
gpshield --configs cartpole --gp.sparse True --seed 1
```

Named blocks in `configs.yaml` (`cartpole`, `mountain_car`, `obstacle`, `obstacle2`, `obstacle3`, `obstacle4`, `hopper`, `road`, `road_2d`, and the `cartpole_v2` / `cartpole_v3` variants) are applied on top of `defaults`.

### Logging

```bash
gpshield --configs cartpole --tensorboard True --logdir ./logdir/cartpole
tensorboard --logdir ./logdir
```

### Recording videos

Cartpole and mountain car render RGB frames. Pass `--record_video True` to wrap the training and evaluation environments with Gymnasium's `RecordVideo` wrapper. Videos are written under `./videos/<env_id>/<reward_id>/`.

## Tests

With uv:

```bash
uv sync --group dev
uv run pytest -v
```

With conda or pip, install the project, then pytest, and run from the repository root:

```bash
pip install -e .
pip install "pytest>=8"
pytest -v
```

The suite checks that the scientific stack imports, that the custom environments follow the Gymnasium `reset` and `step` contract, and that model construction plus a tiny training job can start.

## Dependency sync

`requirements.txt` must match `[project.dependencies]` in `pyproject.toml`. CI runs `scripts/check_requirements_sync.py` when either file changes. Locally:

```bash
python scripts/check_requirements_sync.py
```
