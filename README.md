# Safe RL with GP

This repo contains the code for the experimental evaluation in the research paper "Safe Reinforcement Learning via Recovery-based Shielding with Gaussian Process Dynamics Models"

## Dependencies

The main dependencies of the project are ```gym==0.25.2```, ```numpy>=1.26.4```, ```jax==0.4.30``` and ```gpjax==0.8.2```. We provide a requirements.txt and environment.yml, we recommend first setting up a conda environment for this project
```
conda create -n gpjax --clone base
```
The specific JAXLib installation you require may be different from ours as it is hardware dependent, see (https://github.com/google/jax#pip-installation-gpu-cuda) for details. Otherwise you can install the requirements with pip or similar.

The docs for the GPJAX library can be found here (https://docs.jaxgaussianprocesses.com/) we recommend installing ```gpjax==0.8.2``` but our code may be compatible with newer versions.

```
conda activate gpjax
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
```
Finally for MuJoCo dependencies, please install,
```
pip install gymnasium[mujoco]
```

## Running experiments

We use yaml files for configuring runs, for example to run the experiment for cartpole simple run the following in the command line,
```
python run.py --configs cartpole
```
Configs can easily be overridden by pass them as command line arguments
```
python run.py --configs cartpole --gp.sparse True
```
By editing the configs.yaml file you can also setup a set of default configs.

## Plotting

For plotting runs our code uses tensorboard. Simply add ```--tensorboard True --logdir my_logdir/``` to the command line when running experiments. Then launch tensorboard.
```
tensorboard --logdir my_logdir
```

## Recording videos

We also support video recording for a limited subset of the environments. Simply add ```--record_video True``` to the command line when running experiments.

