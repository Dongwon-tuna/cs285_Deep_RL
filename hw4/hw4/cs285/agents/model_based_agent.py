from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu


class ModelBasedAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.ob_dim,
                    self.ac_dim,
                )
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=ptu.device)
        )

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
        """

        self.update_statistics(obs, acs, next_obs)
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        # TODO(student): update self.dynamics_models[i] using the given batch of data
        # HINT: make sure to normalize the NN input (observations and actions)

        normalized_obs_acs = (torch.cat([obs, acs], dim=-1) - self.obs_acs_mean)/self.obs_acs_std
        normalized_obs_delta = ((next_obs - obs) - self.obs_delta_mean)/self.obs_delta_std
        # *and* train it with normalized outputs (observation deltas) 
        # HINT 2: make sure to train it with observation *deltas*, not next_obs
        # directly
        model = self.dynamics_models[i]
        predicted_model = model(normalized_obs_acs)


        # HINT 3: make sure to avoid any risk of dividing by zero when
        # normalizing vectors by adding a small number to the denominator!
        loss = self.loss_fn(normalized_obs_delta , predicted_model) # 두개 인자 빼서 넣어주지 않음 그냥 각각 넣어줌

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 각각의 앙상블 하나는 여기서 학습하고 Run 파일에서는 이 로스를 평균내서 그냥 Dynamics model의 평균 성능을 보는거..

        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        # TODO(student): update the statistics
        obs_acs = torch.cat([obs, acs], dim=-1)
        self.obs_acs_mean = obs_acs.mean(dim=0)
        self.obs_acs_std = obs_acs.std(dim=0) + 1e-8 

        obs_delta = next_obs - obs
        self.obs_delta_mean = obs_delta.mean(dim=0)
        self.obs_delta_std = obs_delta.std(dim=0) + 1e-8

    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        # 1. 모델 입력(obs+acs) 정규화
        normalized_obs_acs = (torch.cat([obs, acs], dim=-1) - self.obs_acs_mean) / (self.obs_acs_std)
        
        # 2. 모델 예측값(정규화된 obs_delta) 얻기
        model = self.dynamics_models[i]
        predicted_obs_delta_normalized = model(normalized_obs_acs)
        
        # 3. 모델 예측값(obs_delta) 비정규화
        # obs_delta의 통계값 사용  => 여기서 좀 헷갈렸음. 우리가 학습했던 진짜 값을 가지고 unnormalized 해줘야함
        unnormalized_obs_delta = predicted_obs_delta_normalized * self.obs_delta_std + self.obs_delta_mean
        
        # 4. 다음 관측값(next_obs) 예측
        pred_next_obs = obs + unnormalized_obs_delta 

        # TODO(student): get the model's predicted `next_obs`
        # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        return ptu.to_numpy(pred_next_obs)

    # def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
    #     """
    #     Evaluate a batch of action sequences using the ensemble of dynamics models.

    #     Args:
    #         obs: starting observation, shape (ob_dim,)
    #         action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
    #     Returns:
    #         sum_of_rewards: shape (mpc_num_action_sequences,)
    #     """
    #     # We are going to predict (ensemble_size * mpc_num_action_sequences)
    #     # distinct rollouts, and then average over the ensemble dimension to get
    #     # the reward for each action sequence.

    #     # We start by initializing an array to keep track of the reward for each
    #     # of these rollouts.
    #     sum_of_rewards = np.zeros(
    #         (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
    #     )
    #     # We need to repeat our starting obs for each of the rollouts.
    #     obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

    #     # TODO(student): for each batch of actions in in the horizon...
    #     for acs in range(self.mpc_horizon):
    #         assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
    #         assert obs.shape == (
    #             self.ensemble_size,
    #             self.mpc_num_action_sequences,
    #             self.ob_dim,
    #         )
    #         for i in range(self.ensemble_size) :
    #             predicted_obs = self.get_dynamics_predictions(i,obs[i],action_sequences)
    #             next_obs.append(predicted_obs)
    #         # TODO(student): predict the next_obs for each rollout
    #         # HINT: use self.get_dynamics_predictions


    #         next_obs = np.array(next_obs)
    #         assert next_obs.shape == (
    #             self.ensemble_size,
    #             self.mpc_num_action_sequences,
    #             self.ob_dim,
    #         )

    #         # TODO(student): get the reward for the current step in each rollout
    #         # HINT: use `self.env.get_reward`. `get_reward` takes 2 arguments:
    #         # `next_obs` and `acs` with shape (n, ob_dim) and (n, ac_dim),
    #         # respectively, and returns a tuple of `(rewards, dones)`. You can 
    #         # ignore `dones`. You might want to do some reshaping to make
    #         # `next_obs` and `acs` 2-dimensional.

    #         rewards = self.env.get_reward()
    #         assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

    #         sum_of_rewards += rewards

    #         obs = next_obs

    #     # now we average over the ensemble dimension
    #     return sum_of_rewards.mean(axis=0)

    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        # We are going to predict (ensemble_size * mpc_num_action_sequences)
        # distinct rollouts, and then average over the ensemble dimension to get
        # the reward for each action sequence.

        # We start by initializing an array to keep track of the reward for each
        # of these rollouts.
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        # We need to repeat our starting obs for each of the rollouts.
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        # TODO(student): for each batch of actions in in the horizon...
        for h in range(self.mpc_horizon):
            acs = action_sequences[:, h, :]
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )
            next_obs_list = []
            # TODO(student): predict the next_obs for each rollout
            # HINT: use self.get_dynamics_predictions
            for i in range(self.ensemble_size):
                next_obs_list.append(self.get_dynamics_predictions(i, obs[i], acs))

            next_obs = np.array(next_obs_list)
            
            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # TODO(student): get the reward for the current step in each rollout
            # HINT: use `self.env.get_reward`. `get_reward` takes 2 arguments:
            # `next_obs` and `acs` with shape (n, ob_dim) and (n, ac_dim),
            # respectively, and returns a tuple of `(rewards, dones)`. You can 
            # ignore `dones`. You might want to do some reshaping to make
            # `next_obs` and `acs` 2-dimensional.

            # get_reward구조에 맞추기 위해서 Ensemble 갯수와 action seq 크기 곱해서 N으로 맞춰줄 필요 있음

            resizeN = self.mpc_num_action_sequences * self.ensemble_size
            re_next_obs = next_obs.reshape(resizeN, self.ob_dim)
            re_acs = np.tile(acs, (self.ensemble_size, 1))

            rewards,done = self.env.get_reward(re_next_obs,re_acs)

            rewards = rewards.reshape(self.ensemble_size, self.mpc_num_action_sequences)
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)    

    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        action_sequences = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
        )#동작에서의 최대값, 최소값 가져오고 그 값 사이에서 랜덤생성
        #이미 mpc num 갯수만큼 호라이즌*aDim이 생김.

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            #breakpoint()
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return action_sequences[best_index][0]
        elif self.mpc_strategy == "cem":
            # Initialize the distribution parameters once before the loop
            elite_mean = np.zeros((self.mpc_horizon, self.ac_dim))
            elite_std = np.ones((self.mpc_horizon, self.ac_dim))

            # The comment is not literally included in the code as it would be a syntax error.
            # However, the logic for the special case of i==0 is correctly implemented here:
            # The initial elite_mean and elite_std (zeros and ones) are used to sample
            # the action sequences for the very first iteration. This handles the
            # "special case" logic correctly without needing an explicit if statement inside the loop.
            
            for i in range(self.cem_num_iters):
                # Sample action sequences from the current distribution
                action_sequences = np.random.normal(loc=elite_mean, scale=elite_std, size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim))

                rewards = self.evaluate_action_sequences(obs, action_sequences)

                elite_indices = np.argsort(rewards)[-self.cem_num_elites:]
                elite_action_sequences = action_sequences[elite_indices]

                new_elite_mean = np.mean(elite_action_sequences, axis=0)
                new_elite_std = np.std(elite_action_sequences, axis=0)

                elite_mean = (1 - self.cem_alpha) * elite_mean + self.cem_alpha * new_elite_mean
                elite_std = (1 - self.cem_alpha) * elite_std + self.cem_alpha * new_elite_std
            

            return np.squeeze(elite_mean[0]).astype(np.float32)


                
        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
