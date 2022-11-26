import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import cv2
from RL_Actions import *

def cvimage_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pygame.image.frombuffer(image.tostring(), image.shape[1::-1], "BGR")

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None,size=(512,512,3)):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=self.size, dtype=np.uint8)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(10)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"image": self._image}

    def _get_info(self):
        return {
            "last action": self._last_action
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        # here we should read a new image from the dataset
        img = cv2.imread("city2_hazy.png")
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self._image=cv2.resize(img,(512,512))
        self._last_action=-1

        # # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        print(action)
        # (img*255).astype(np.uint8)
        if action==0 :
            self._image=dehazeDCP(self._image)
        elif action==1:
            self._image=dehazeCA(self._image)
        elif action ==2 :
            self._image=gammaCorrection(self._image, 1.5)
        elif action ==3:
            self._image=cv2.cvtColor(ChannelValue(self._image,1.05,0),cv2.COLOR_RGB2BGR)
        elif action ==4:
            self._image=cv2.cvtColor(ChannelValue(self._image,1.05,1),cv2.COLOR_RGB2BGR)
        elif action ==5:
            self._image=cv2.cvtColor(ChannelValue(self._image,1.05,2),cv2.COLOR_RGB2BGR)
        elif action ==6:
            self._image=cv2.cvtColor(ChannelValue(self._image,0.95,0),cv2.COLOR_RGB2BGR)
        elif action ==7:
            self._image=cv2.cvtColor(ChannelValue(self._image,0.95,1),cv2.COLOR_RGB2BGR)
        elif action ==8:
            self._image=cv2.cvtColor(ChannelValue(self._image,0.95,2),cv2.COLOR_RGB2BGR)
        elif action ==9:
            self._image=cv2.cvtColor(CE(self._image),cv2.COLOR_RGB2BGR)
        
        self._last_action=action
        # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        terminated=np.random.uniform()
        reward = 1 if terminated<0.25 else 0  # Random Reward
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated<0.25, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        imp=cvimage_to_pygame(self._image)
        canvas.blit(imp, (0, 0))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
