# A2C-GP-Shield

The code repository for **Safe reinforcement learning via recovery-based shielding with Gaussian Process dynamics models.**. Full details can be found in:

> **Safe reinforcement learning via recovery-based shielding with Gaussian Process dynamics models.**
> Alexander W. Goodall, Francesco Belardinelli
> arXiv: [https://arxiv.org/abs/2602.12444](https://arxiv.org/abs/2602.12444)

*Disclaimer:* This version of the code uses a newer software stack (`gpjax==1.0.0`, `jax>=0.7.0`, `numpy>=2.0.0`) compared to the code used to collect the results reported in teh paper, the results therefore may not be fully repoduceable on a per-seed basis. 

## Software stack

Training learns a Gaussian Process dynamics model with [GPJAX](https://docs.jaxgaussianprocesses.com/), runs the model and the actor-critic on CPU or CUDA GPUs with [JAX](https://docs.jax.dev/), and steps through the environments with [Gymnasium](https://gymnasium.farama.org/).

## Quick Start

### Installation

#### Prerequisites

Python 3.12 is required.

#### Installation with conda

- Install conda, e.g., via [anaconda](https://anaconda.org/channels/anaconda/packages/conda/overview).
- Clone the repo:

```bash
git clone https://github.com/sacktock/A2C-GP-Shield.git
```

- Create a conda virtual environment:

```bash
conda env create --name gpshield --file conda-environment.yaml
conda activate gpshield
```

- Install the project and its dependencies:

```bash
pip install -e .
```

> `pip install -e .` installs the dependencies declared in `pyproject.toml`. `requirements.txt` lists the same base dependencies and is kept in lockstep with that file.

#### Installation with uv

From the repository root, `uv sync` installs the base dependency set (CPU JAX):

```bash
uv sync
```

### Enabling GPU acceleration with JAX (optional)

GP-Shield uses [JAX](https://docs.jax.dev/) for the dynamics model and the actor-critic. CPU is enough to smoke test the environments and run a short training job.

**macOS and Windows:** we recommend JAX with CPU. No further action is required if you correctly followed the earlier steps.

#### pip

**Linux x86_64/aarch64** and **Windows WSL2 x86_64**: We strongly recommend using [Ubuntu 22.04](https://apps.microsoft.com/detail/9pn20msr04dw?hl=en-GB&gl=BE) or similar, jax and jaxlib should already be installed. You need to reinstall [JAX](https://docs.jax.dev/) based on your cuda driver compatibility. Do not use the `-U` option here as it may break earlier dependencies!

```bash
pip install "jax[cuda13]"
```

```bash
pip install "jax[cuda12]"
```


#### uv

Alternatively with uv. These groups follow the JAX version in the lock, so they stay compatible with `gpjax==1.0.0` (`jax<0.11`).

```bash
uv sync --group cuda12
```

```bash
uv sync --group cuda13
```

### Extras

MuJoCo (used by Hopper) is installed through the `gymnasium[mujoco]` extra. On a headless Linux machine (CI job), also install OSMesa and select it, or `Hopper-v5` fails while importing its renderer:

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

### Logging and Miscellaneous

**TensorBoard** logging is supported by our implementation and is enabled in the following example:

```bash
gpshield --configs cartpole --tensorboard True --logdir ./logdir/cartpole
```

The TensorBoard logs can be accessed via the command line:

```bash
tensorboard --logdir ./logdir/cartpole
```

#### Recording videos

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

The test suite is small and checks that the scientific stack imports, that the custom environments follow the Gymnasium `reset` and `step` contract, and that model construction plus a tiny training job can start.

## Dependency sync

`requirements.txt` must match `[project.dependencies]` in `pyproject.toml`. CI runs `scripts/check_requirements_sync.py` when either file changes. Locally:

```bash
python scripts/check_requirements_sync.py
```


## License

Our implementation is released under the MIT License.