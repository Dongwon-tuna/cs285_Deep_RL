import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        return self.network(obs)
        


    #########################################
    #여기에선 V를 학습시켜주는 것. 실제 traj에서 얻은 q리턴을 가지고 있으니 그것을 목표로 잡고 현재 critic이 예측하는 v를
    #q에 가까워지도록 하는 것.
    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        #q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-8)

        # TODO: update the critic using the observations and q_values
        predicted_values = self(obs)
        #loss = F.mse_loss(predicted_values, q_values)
        loss = F.mse_loss(predicted_values.squeeze(), q_values)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }