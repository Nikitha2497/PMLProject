from gymnasium.envs.registration import register

register(
     id="env/DehazeAgent-v0",
     entry_point="env:DehazeAgent",
     max_episode_steps=300,
)