import gym

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
obs = env.reset()
print("환경 초기화 완료!")

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"[Step {step}] Action: {action}, Reward: {reward}, Done: {done}")
    
    if done:
        print("에피소드 끝! 환경 초기화")
        obs = env.reset()

env.close()
