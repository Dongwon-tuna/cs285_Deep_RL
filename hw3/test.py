import gym

# Atari 환경 로드
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
obs = env.reset()

print("환경 초기화 완료!")
for _ in range(100):
    obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        obs = env.reset()

env.close()
