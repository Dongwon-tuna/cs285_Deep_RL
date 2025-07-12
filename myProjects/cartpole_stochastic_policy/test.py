import torch
import torch.nn as nn
import torch.nn.functional as F
import gym


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA GPU available.")


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


env = gym.make("CartPole-v1", render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

policy = MLPPolicy(obs_dim, act_dim).to(device)

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = policy(obs_tensor)
        action = torch.argmax(logits, dim=-1).item()

    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

env.close()
print("Total Reward:", total_reward)
