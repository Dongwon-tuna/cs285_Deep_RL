

# HW 4



## Note
- Implemented algorithms:
  - [x] Model Predictive Control (MPC)
  - [x] Cross-Entropy Method (CEM)
  - [x] Model-Based Policy Optimization (MBPO)


## Analysis

---
### Problem 2.1



![ ](imgs/analysis.png)
![ ](imgs/analysis_2.1.jpeg)

### Problem 2.2
![ ](imgs/analysis_2.2.jpeg)

## Code

---
### Problem 1

```
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_0_iter.yaml
```
experiments 1
num_layers: 1, hidden_size: 32

![ ](imgs/1.1.png)

experiments 2
num_layers: 2, hidden_size: 64

![ ](imgs/1.2.png)

experiments 3
num_layers: 3, hidden_size: 128

![ ](imgs/1.3.png)


### Problem 2
```
python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_1_iter.yaml
```
![ ](imgs/2.png)


### Problem 3
Experiment 1
```
python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_multi_iter.yaml
```
![status](https://img.shields.io/badge/experiment1-lightgrey)

![ ](imgs/3.1_dynamics.png)
![ ](imgs/3.1_return.png)

Experiment 2
```
python cs285/scripts/run_hw4.py -cfg experiments/mpc/reacher_multi_iter.yaml
```
![status](https://img.shields.io/badge/experiment2-orange)

![ ](imgs/3.2_dynamics.png)
![ ](imgs/3.2_return.png)

Experiment 3
```
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_multi_iter.yaml
```
![status](https://img.shields.io/badge/experiment3-blue)

![ ](imgs/3.3_dynamics.png)
![ ](imgs/3.3_return.png)

### Problem 4

```
python cs285/scripts/run_hw4.py -cfg experiments/mpc/reacher_ablation.yaml
```
Effect of the number of candidate action sequences
![status](https://img.shields.io/badge/mpc_num_action_sequences=500-pink)
![status](https://img.shields.io/badge/mpc_num_action_sequences=1000-red)
![status](https://img.shields.io/badge/mpc_num_action_sequences=1500-9cf)

![ ](imgs/4.1_dynamics.png)
![ ](imgs/4.1_return.png)

**observed trends :** A larger number of action sequences generally improves performance by increasing the chance of finding a good plan.

Effect of planning horizon
![status](https://img.shields.io/badge/mpc_horizon=5-lightgrey)
![status](https://img.shields.io/badge/mpc_horizon=10-red)
![status](https://img.shields.io/badge/mpc_horizon=15-greenblue)

![ ](imgs/4.2_dynamics.png)
![ ](imgs/4.2_return.png)

**observed trends :** The ideal planning horizon is a trade-off: it must be long enough to capture meaningful rewards but short enough to avoid cumulative prediction errors.


Effect of ensemble size
![status](https://img.shields.io/badge/ensemble_size=1-lightgrey)
![status](https://img.shields.io/badge/ensemble_size=3-pink)
![status](https://img.shields.io/badge/ensemble_size=5-9cf)

![ ](imgs/4.3_dynamics.png)
![ ](imgs/4.3_return.png)

**observed trends :** “The reward was higher with ensemble sizes of 3 or 5 compared to a single model, but beyond 3 ensembles, additional models did not provide significant benefits.”

### Problem 5

```
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_cem.yaml
```
Effect of CEM iteration
![status](https://img.shields.io/badge/cem_iterations=2-pink)
![status](https://img.shields.io/badge/cem_iterations=4-orange)

![ ](imgs/5.1_dynamics.png)
![ ](imgs/5.1_return.png)

**observed trends :**  Increasing the number of CEM iterations from 2 to 4 improves performance because the algorithm has more opportunities to refine the action distribution. Each additional iteration allows the model to select a more elite set of candidates, leading to a more precisely focused distribution for subsequent sampling and ultimately a better plan.

![status](https://img.shields.io/badge/cem_iterations=2-pink)
![status](https://img.shields.io/badge/cem_iterations=4-orange)
![status](https://img.shields.io/badge/random_shooting-blue)

![ ](imgs/5.2_dynamics.png)
![ ](imgs/5.2_return.png)

**observed trends :**  CEM is superior to random shooting because it iteratively refines the search space, allowing it to focus on areas with a higher density of good actions rather than searching randomly across the entire action space.

### Problem 6

```
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_mbpo.yaml --sac_config_file experiments/sac/halfcheetah_clipq.yaml
```
Effect of MBPO rollout length

![status](https://img.shields.io/badge/mbpo_rollout_length=0-orange)
![status](https://img.shields.io/badge/mbpo_rollout_length=1-blue)
![status](https://img.shields.io/badge/mbpo_rollout_length=10-brightgreen)

![ ](imgs/6_dynamics.png)
![ ](imgs/6_data.png)
![ ](imgs/6_return.png)

**observed trends :**  The advantage of a longer rollout length is that it increases the amount of model-generated data, leading to a more stable and accurate actor and critic. 
