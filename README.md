# SARL
State Abstractions for Reinforcement Learning.

## Dependencies
Install the following dependencies to be able to run the code, preferably in a virtual environment (with python3.6 or higher).
```
pip install tensorflow==1.15.0
pip install tensorflow-probability==0.8.0
pip install tf-agents==0.3.0
pip install numpy
pip install gym==0.12.0
pip install torch torchvision
pip install IPython
pip install matplotlib
pip install opencv-python
```
Mujoco physics simulator is required to run ant experiments. Follow instructions [here](https://github.com/openai/mujoco-py) to install it.

## Instructions

### Rooms environments
Make sure that the current directory contains the package `sarl`.  The command to run the algorithm (A_AVI) in the paper for
the rooms environments is:
```
python -m sarl.examples.rooms_dist -e {environment_number} -n {run_number} -d {directory} -p 1
```

 - `environment_number` is either 2 (9-Rooms) or 4 (16-Rooms).
 - `run_number` is any integer, used for naming the files generated during different executions.
 - `directory` is the path to a directory for output files. Use different directory for different environments but same directory for all executions.
 - Use option '-p' if you want to train the options in parallel.

After running K complete executions (`run_number` varying from 0 to K), plot the learning curve by running:
```
python -m sarl.scripts.single_plot -d {directory} -n K
```
The command for the ablation (no alternation) is:
```
python -m sarl.examples.rooms_dist -e {environment_number} -n {run_number} -d {directory} -p 1 -b
```
The command for running with different choices of abstract states is
```
python -m sarl.examples.rooms_dist -e {environment_number} -n {run_number} -d {directory} -p 1 -r {abstract_states}
```
where `abstract_states` is 1 (full_rooms) or 2 (room_centers).

### Ant environments
The training for ant environments is similar:
```
python -m sarl.examples.ant_dist -e {environment_number} -n {run_number} -d {directory} -a td3
```
where `environment_number` is either 0 (AntMaze), 1 (AntPush) or 2 (AntFall). The option `-a td3` uses TD3 to learn the
options instead of ARS (default). Use option `-b` for ablation. 

### Spectrl baseline
The command for rooms environments is:
```
python -m spectrl.examples.rooms -e {environment_number} -d {directory} -n {run_number}
```
where the options are the same as before.

### Efficient HRL baseline
This baseline depends on tensorflow version 1.14.0 and tf-agents version 0.2.0rc2.
To run the HIRO baseline for rooms environments, change directory to `sarl/efficient_hrl`:

```
cd efficient_hrl
```
To run HIRO-ORIG:
```
tensorboard --logdir {directory}/eval{run_number} & python scripts/local_eval.py {run_number} hiro_orig rooms{2+environment_number} rooms_uvf {directory} & python scripts/local_train.py {run_number} hiro_orig rooms{2+environment_number} rooms_uvf {directory}
```
To run the baseline for ant environments, the command is similar, except replace `rooms{2+environment_number}` with one of the following: `ant_maze`, `ant_push_multi` or `ant_fall_multi`. Also replace `rooms_uvf` with `base_uvf`.
