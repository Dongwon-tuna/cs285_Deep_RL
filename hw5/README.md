

# HW 5

## Note

- Implemented algorithms:
  - [x] Random Network Distillation (RND)
  - [x] Conservative Q-Learning (CQL)
  - [x] Advantage Weighted Actor Critic (AWAC)
  - [x] Implicit Q-Learning (IQL)



## Analysis

---


![ ](imgs/analysis1.png)

### Problem 1.1

![ ](imgs/analysis1.1.png)

![ ](imgs/1.1.jpeg)

### Problem 1.2

![ ](imgs/analysis1.21.png)

![ ](imgs/analysis1.22.png)

![ ](imgs/1.2.jpeg)

### Problem 1.3
![ ](imgs/analysis1.3.png)

![ ](imgs/1.3.jpeg)

## Code

---

## 3 Exploration  

`total_steps = 10000`

Running a random policy
```
python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_easy_random.yaml --dataset_dir datasets/


python cs285/scripts/run_hw5_explore.py  -cfg experiments/exploration/pointmass_medium_random.yaml  --dataset_dir datasets/


python cs285/scripts/run_hw5_explore.py  -cfg experiments/exploration/pointmass_hard_random.yaml --dataset_dir datasets/
```

Random Network Distillation
```
python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_easy_rnd.yaml --dataset_dir datasets/

python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_medium_rnd.yaml  --dataset_dir datasets/

python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_hard_rnd.yaml --dataset_dir datasets/
```


|            | Easy                          | Medium                          | Hard                          |
|------------|-------------------------------|---------------------------------|-------------------------------|
| Random     | ![](imgs/3.1_easy.png)     | ![](imgs/3.1_medium.png)      | ![](imgs/3.1_hard.png)     |
| RND        | ![](imgs/3.2_easy.png)     | ![](imgs/3.2_medium.png)      | ![](imgs/3.2_hard.png)     |


---
## 4 Offline RL 



### 4.1 CQL 

```
python ./cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_easy_cql.yaml --dataset_dir datasets

python ./cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_easy_dqn.yaml --dataset_dir datasets

python ./cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_medium_cql.yaml --dataset_dir datasets

python ./cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_medium_dqn.yaml --dataset_dir datasets

python ./cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_hard_cql.yaml --dataset_dir datasets

python ./cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_hard_dqn.yaml --dataset_dir datasets
```

### Easy
![status](https://img.shields.io/badge/cql-red)  ![status](https://img.shields.io/badge/dqn-blue)
|            | Easy                          
|------------|------------------------------
| len    | ![](imgs/4.1_easy_len.png)     
| return | ![](imgs/4.1_easy_return.png)  


![status](https://img.shields.io/badge/cql-red)
![](imgs/4.1_easy_cql.png) 
![status](https://img.shields.io/badge/dqn-blue)
![](imgs/4.1_easy_dqn.png) 


### Medium
![status](https://img.shields.io/badge/cql-red)  ![status](https://img.shields.io/badge/dqn-green)
|            | Medium                          
|------------|------------------------------
| len    | ![](imgs/4.1_medium_len.png)     
| return | ![](imgs/4.1_medium_return.png)  


![status](https://img.shields.io/badge/cql-red)
![](imgs/4.1_medium_cql.png) 
![status](https://img.shields.io/badge/dqn-green)
![](imgs/4.1_medium_dqn.png) 


### Hard
![status](https://img.shields.io/badge/cql-lightgrey)  ![status](https://img.shields.io/badge/dqn-orange)
|            | Hard                         
|------------|------------------------------
| len    | ![](imgs/4.1_hard_len.png)     
| return | ![](imgs/4.1_hard_return.png)  


![status](https://img.shields.io/badge/cql-lightgrey)
![](imgs/4.1_hard_cql.png) 
![status](https://img.shields.io/badge/dqn-orange)
![](imgs/4.1_hard_dqn.png)  


### α
On the Medium environment, create several experiment variations in which the value of the α parameter is varied, from α = 0 (equivalent to DQN) to α = 10.
I try α = 0, 0.2 , 1, 5, 10. 



|            | ![status](https://img.shields.io/badge/α=0-blue)  ![status](https://img.shields.io/badge/α=0,2-orange) ![status](https://img.shields.io/badge/α=0,5-red)  ![status](https://img.shields.io/badge/α=1-cyan) ![status](https://img.shields.io/badge/α=5-pink) ![status](https://img.shields.io/badge/α=10-green)                      
|------------|------------------------------
| len    | ![](imgs/4.1_alpha_len.png)     
| return | ![](imgs/4.1_alpha_return.png)  

Conclusion: A moderately small α (e.g., 0.1–0.5) often works best, as it effectively suppresses OOD overestimation while maintaining stable performance.

---

### 4.2 Policy Constraint Methods: IQL and AWAC


#### Comparison: AWAC vs IQL

| Method | Pros | Cons |
|--------|------|------|
| **AWAC** | Intuitive and relatively simple to implement. | If advantage estimation is inaccurate, the policy may be updated with incorrect weights → potential performance degradation. <br> Strongly dependent on dataset quality. |
| **IQL** | Reduces instability since the actor is not explicitly trained. <br> Leverages expectile regression to stably estimate the Q–V gap → highly effective on large-scale offline datasets. | Implementation is more complex and requires tuning of additional hyperparameters (e.g., expectile τ). |



|            | ![status](https://img.shields.io/badge/IQL-blue)  ![status](https://img.shields.io/badge/AWAC-orange)                 
|------------|------------------------------
| len    | ![](imgs/4.2_len.png)     
| return | ![](imgs/4.2_return.png)  


![status](https://img.shields.io/badge/IQL-blue)
![](imgs/4.2_iql.png) 
![status](https://img.shields.io/badge/AWAC-orange) 
![](imgs/4.2_awac.png) 


### 4.3 Data ablations

Run in Hard environment

When the maze becomes more challenging, the agent struggles to reach the goal with offline training alone. However, increasing the dataset size can significantly improve performance.

|            | ![status](https://img.shields.io/badge/1000-lightgrey)  ![status](https://img.shields.io/badge/5000-orange) ![status](https://img.shields.io/badge/10000-blue) ![status](https://img.shields.io/badge/20000-red)  
|------------|------------------------------
| len    | ![](imgs/4.3_len.png)     
| return | ![](imgs/4.3_return.png)  



| total_steps = 1000 | total_steps = 5000 | total_steps = 10000 | total_steps = 20000 |
|--------------------|--------------------|---------------------|---------------------|
| ![](imgs/4.3_1000.png) | ![](imgs/4.3_5000.png) | ![](imgs/4.3_10000.png) | ![](imgs/4.3_20000.png) |


---

### 5 Online Fine-Tuning


The agent was trained offline up to 100k steps, followed by online fine-tuning until 200k steps. Notably, both IQL and AWAC showed performance improvements after 100k steps.

|            | ![status](https://img.shields.io/badge/cql-blue)  ![status](https://img.shields.io/badge/awac-red) ![status](https://img.shields.io/badge/iql-green) 
|------------|------------------------------
| len    | ![](imgs/5_len.png)     
| return | ![](imgs/5_return.png)  




|            | Offline Trainning            | Online Finetuning           |
|------------------|------------------|------------------|
|  CQL | ![](imgs/5_cql_off.png)  | ![](imgs/5_cql_on.png)  |
| AWAC | ![](imgs/5_awac_off.png)  | ![](imgs/5_awac_on.png)  |
| IQL | ![](imgs/5_iql_off.png)  | ![](imgs/5_iql_on.png)  |

