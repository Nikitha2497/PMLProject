import gymnasium
from gymnasium.envs.registration import register

register(
     id="dehaze_agent/DehazeAgent2-v0",
     entry_point="dehaze_agent:DehazeAgent2",
     max_episode_steps=300,
)


env = gymnasium.make('dehaze_agent/DehazeAgent2-v0',render_mode='human')
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()