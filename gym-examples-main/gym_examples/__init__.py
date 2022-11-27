from gymnasium.envs.registration import register
import gymnasium

register(
     id="gym_examples/DehazeAgent-v0",
     entry_point="gym_examples.envs:DehazeAgent",
     max_episode_steps=300,
)