import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import cv2
from RL_Actions import *
import math
import pandas as pd

import skimage

import torch
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.dataloaders import LoadImages

# Hyper Parameters for tuning reward function.
# L2 norm loss
lamba1 = 0.001
# PSNR metric measure
lambda2 = 0.1
# SSIM metric measure
lambda3 = 20
# Bounding box loss
lambda4 = 1

def cvimage_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pygame.image.frombuffer(image.tostring(), image.shape[1::-1], "BGR")

class DehazeAgent(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None,size=(512,512,3)):
        self.size = size  # The size of the square grid
        self.window_size = 640  # The size of the PyGame window
        self.dataset_file="yolov5/datasets/cityscapes/train_data.csv"
        self.source_folder="yolov5/datasets/cityscapes/images/train/" # clear images
        self.target_folder="yolov5/datasets/cityscapes/images/train_foggy/" # foggy images
        # self.dataset_file="/Users/iamariyap/Desktop/sem3/PredictiveML/RL_Project/code/PMLProject/src/yolov5/datasets/cityscapes/train_data.csv"
        # self.source_folder="/Users/iamariyap/Desktop/sem3/PredictiveML/RL_Project/code/PMLProject/src/yolov5/datasets/cityscapes/images/train/" # clear images
        # self.target_folder="/Users/iamariyap/Desktop/sem3/PredictiveML/RL_Project/code/PMLProject/src/yolov5/datasets/cityscapes/images/train_foggy/" # foggy images
        self.df=pd.read_csv(self.dataset_file,index_col='Unnamed: 0.1')
        self.model=torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt', source='local',autoshape=False)
        # self.model=torch.hub.load('/Users/iamariyap/Desktop/sem3/PredictiveML/RL_Project/code/PMLProject/src/yolov5', 'custom', path='/Users/iamariyap/Desktop/sem3/PredictiveML/RL_Project/code/PMLProject/src/yolov5/runs/train/exp/weights/best.pt', source='local',autoshape=False)

        self.model.warmup(imgsz=(1 , 3, 640, 640))
        self.loss_function=ComputeLoss(self.model.model)
        self.count=0
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=self.size, dtype=np.uint8)
            }
        )
 
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
        return {"image": self._image , "path": self._original_path}

    def _get_info(self):
        return {
            "last action": self._last_action
        }
    
    def _l2_norm(self):
        return math.sqrt(np.sum((self._image - self._original_clear_img)**2))

    def _pnsr_measure(self):
        return skimage.metrics.peak_signal_noise_ratio(self._original_clear_img, self._image)
    
    def _ssim_measure(self):
        return skimage.metrics.structural_similarity(self._original_clear_img, self._image, multichannel = True)
    
    def _bb_loss(self):
        img = self._image
        # format image as required for model
        img=np.copy(img)
        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).to(self.model.device)
        img =  img.float()
        img=img[None]
        # print(img.shape)
        pred = self.model(img,augment=False,visualize=False)

        # obtain the annotations
        lines=[]
        annotation=self._datapoint['annotation'].item()
        annotation_file_name=annotation.split('_foggy')[0].split('target/')[-1]+'.txt'
        annotation_file_name='yolov5/datasets/cityscapes/labels/train/'+annotation_file_name
        # annotation_file_name='/Users/iamariyap/Desktop/sem3/PredictiveML/RL_Project/code/PMLProject/src/yolov5/datasets/cityscapes/labels/train/'+annotation_file_name
        print(annotation_file_name)
        with open(annotation_file_name) as f:
            lines=[line.split(' ') for line in f.readlines()]
        
        # parse to obtain tensor format
        targets=[]
        for line in lines:
            line[-1]=line[-1].split('\n')[0]
            values=[0]
            for val in line:
                values.append(float(val))

            targets.append(torch.tensor([values]))

        all_targets=torch.cat(targets)

        # loss function invocation
        loss_value,_=self.loss_function(pred[1],all_targets.to(self.model.device))

        return loss_value.item()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        # here we should read a new image from the dataset
        self._datapoint=self.df.sample()
        img = cv2.imread(self.target_folder+self._datapoint['foggy_image'].item())
        print(self.target_folder+self._datapoint['foggy_image'].item())
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self._original_size=img.shape
        self._image=cv2.resize(img,(640,640))

        self._original_path = self.source_folder+self._datapoint['src_image'].item()
        print(self._original_path)
        self._original_clear_img = cv2.imread(self._original_path)
        self._original_clear_img = cv2.resize(self._original_clear_img, (640, 640))

        self._last_action=-1
        self.count=0

        # # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

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
        self.count=self.count+1
        # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # terminated=np.random.uniform()
        terminated = False
        bb_loss=self._bb_loss()
        reward = -lamba1*self._l2_norm() + lambda2*self._pnsr_measure() + lambda3*self._ssim_measure() + lambda4*bb_loss
        print("L2 Norm - ", self._l2_norm())
        print("psnr measure - ", self._pnsr_measure())
        print(" ssim measure - ", self._ssim_measure())
        print("bb_loss - ",bb_loss)
        # If the image is very close to the original image then terminate
        if(self._ssim_measure() > 0.99 or self.count>10):
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, False, info

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

    