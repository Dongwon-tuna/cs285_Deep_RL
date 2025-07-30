import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(ptu.device)  #pytorch사용위해서 텐서로,,
        obs_tensor = obs_tensor.unsqueeze(0)  # shape: [1, ob_dim] => MLP입력 넣어주기위해 행렬 형태로
        dist = self.forward(obs_tensor)
        action = dist.sample()
        action = action.cpu().numpy()
        action = action.squeeze(0)


        return action

    def forward(self, obs: torch.FloatTensor):#분포객체를 반환해야 행동샘플링 등 다양한거 가능
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            return torch.distributions.Categorical(logits=logits)

        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            # mean = self.mean_net(obs)
            # std = torch.exp(self.logstd)
            # return torch.distributions.Normal(mean, std)
            mean = self.mean_net(obs)                       # (B, act_dim)
            std = torch.exp(self.logstd).expand_as(mean)    # (B, act_dim)로 확장
            return torch.distributions.Normal(mean, std)
        

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        dist = self.forward(obs)  # 정책 분포 계산 여기에서는 obs의 갯수만큼 분포를 만든다

        log_probs = -dist.log_prob(actions)
        if not self.discrete:
            log_probs = log_probs.sum(axis=-1)

        loss = (log_probs * advantages).sum()
        # 각 상태에 대한 log-probability와 그 상태에서의 advantage가 1:1로 대응되며,
        # 그렇게 계산된 전체 손실(loss)을 평균내어 정책을 업데이트하는 것이 목표임

        self.optimizer.zero_grad()
        loss.backward() #policy network들의 파라미터들의 gradient를 구함
        self.optimizer.step() #경사하강법으로 업데이트해서 더 좋은 정책이 되도록 함

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
