from gymnasium.envs.registration import register
import gymnasium

register(
     id="gym_examples/GridWorld-v0",
     entry_point="gym_examples.envs:GridWorldEnv",
     max_episode_steps=300,
)