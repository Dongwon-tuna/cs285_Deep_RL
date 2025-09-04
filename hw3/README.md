

# Run the code

## Note
- Implemented algorithms:
  - [x] Q-Learning
  - [x] Double Q-Learning
  - [x] Soft Actor-Critic (SAC)
  - [x] Clipped Double Q-Learning
  - [x] Randomized Ensembled Double Q-Learning (REDQ)



## 2 Deep Q-Learning

---
### Deliver 1: DQN on CartPole-v1

```
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml
```


Result 

![Eval Average Returm](imgs/2.4.png)
![status](https://img.shields.io/badge/lr:0.001-blue-blue)

CartPole lr to 0.05

![Eval Average Returm](imgs/2.4_lr.png)
![status](https://img.shields.io/badge/lr:0.05-red-red)

![Q value](imgs/2.4_q.png)

![Critic loss](imgs/2.4_critic.png)

Run DQN on CartPole-v1, but change the learning rate to 0.05 (you can change this in the YAML config file). 
What happens to (a) the predicted Q-values, and (b) the critic error? Can you relate this to any topics from class or the analysis section of this homework?

(a) Effect on the predicted Q-values:

(b) Effect on the critic error:


### Deliver 2: DQN on LunarLander-v2

```
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 3
```

![Eval Average Returm](imgs/2.5.png)

### Deliver 3: Double DQN on LunarLander-v2

```
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 3
```

![Eval Average Returm](imgs/2.5_double.png)

### Deliver 4: DQN implementation on the MsPacman-v0

```
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman.yaml
```

![Eval Average Returm](imgs/2.5_pac_eval.png)

![Train Average Returm](imgs/2.5_pac_train.png)



## 3 Continuous Actions with Actor-Critic

---



### Deliver 1: Actor with REINFORCE


```
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/sanity_invertedpendulum_reinforce.yaml
```

![Eval Average Returm](imgs/3.1.3.png)


### Train an agent on HalfCheetah-v4
```
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reinforce1.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reinforce10.yaml
```
![status](https://img.shields.io/badge/REINFORCE1-pink-pink)
![status](https://img.shields.io/badge/REINFORCE10-green-green)

![Eval Average Returm](imgs/3.1.3_half.png)


### Deliver 2: Actor with REPARAMETRIZE
```
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reparametrize.yaml
```
![status](https://img.shields.io/badge/REINFORCE1-pink-pink)
![status](https://img.shields.io/badge/REINFORCE10-green-green)
![status](https://img.shields.io/badge/REPARAMETRIZE-red-red)

![Eval Average Returm](imgs/3.1.4_half.png)


### Run single-Q, double-Q, and clipped double-Q on Hopper-v4
```
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper_doubleq.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper_clipq.yaml
```
![status](https://img.shields.io/badge/DoubleQ-pink-pink)
![status](https://img.shields.io/badge/ClipQ-green-green)
![status](https://img.shields.io/badge/SingleQ-blue-blue)

![Eval Average Returm](imgs/3.1.5_hopper.png)

### Run Humanoid-v4
```
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/humanoid.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/humanoid_doubleq.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/humanoid_clipq.yaml
```
![status](https://img.shields.io/badge/ClipQ-pink-pink)
![status](https://img.shields.io/badge/SingleQ-red-red)
![status](https://img.shields.io/badge/DoubleQ-blue-blue)

![Eval Average Returm](imgs/sec3.1.5.gif)

![Eval Average Returm](imgs/3.1.5_humanoid.png)
