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
print("obs dim: ")
print(obs_dim) # 관측의 차원임   4
print("act dim: ") 
print(act_dim)# 행동의 차원임    2

policy = MLPPolicy(obs_dim, act_dim).to(device)  #GPU로 모델 옮기기

obs, _ = env.reset()
done = False
total_reward = 0

while True:

    # CartPole 환경의 관측값(obs)은 다음과 같은 4개의 요소:
    # obs[0] → cart position         : 카트의 수평 위치 (x 좌표)
    # obs[1] → cart velocity         : 카트의 속도
    # obs[2] → pole angle            : 막대의 각도 (θ), 수직에서 얼마나 기울었는지 (rad 단위)
    # obs[3] → pole angular velocity : 막대의 각속도 (회전 속도)


    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = policy(obs_tensor)
        action = torch.argmax(logits, dim=-1).item()

    print(f"obseravtion: {obs}")  # ← 여기 추가

    obs, reward, done, _, _ = env.step(action)
    total_reward += reward




env.close()
print("Total Reward:", total_reward)
