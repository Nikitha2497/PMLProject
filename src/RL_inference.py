# We run this after running the RL training module.
# The learned policy is used on the hazy images to create a dehazed image.
# The dehazed image is then given as an input to the detector network

import torch
from dqn import DQN
import gymnasium
import matplotlib.pyplot as plt
import cv2
from vgg_features import extract_vgg_features
import os
import _init_
import sys

sys.path.append(os.path.abspath("./yolov5"))

env = gymnasium.make('env/DehazeAgent-v0', render_mode='human').unwrapped

n_actions = env.action_space.n


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)


policy_net.load_state_dict(torch.load("dqn_policy_net.pt"))
policy_net.eval()
# target_net.load_state_dict(torch.load("dqn_target_net.pt"))

max_steps = 5

# Greedy policy is used.
def select_action(state):
    return policy_net(state).max(1)[1].view(1, 1)


def dehaze_image(img):
    observation, info = env.reset()
    done = False
    count = 0
    state = observation["image"]
    while not done or count < max_steps:
        count += 1
        features = extract_vgg_features(state)
        action = select_action(features)
        observation, reward, done, _, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = observation["image"]
        else:
            return state

        # Move to the next state
        state = next_state

    return state

# img = cv2.imread("/Users/yani/Documents/GitHub/PMLProject/src/city2_hazy.png")

# dehazed_img = dehaze_image(img)
# cv2.imwrite( "dehazed_img1.png", dehazed_img)
# # plt.imshow(dehazed_img)
def generate_dehazed_output(input_path,out_path):
    print("Reading images from {} and writing to {}".format(input_path,out_path))
    image_files=os.listdir(input_path)
    for image_name in image_files: 
        img_path=input_path+image_name
        img = cv2.imread(img_path)
        dehazed_img = dehaze_image(img)
        new_img_path=out_path+image_name.split(".")[0]+"_dehazed.png"
        print("Writing image : ",new_img_path)
        cv2.imwrite(new_img_path, dehazed_img)
    
    return

mode= ''# yolo_inference/yolo_train
user_input=input('Enter the mode (yolo_inference/yolo_train) :')

path_={'yolo_inference':['yolov5/datasets/cityscapes/images/test_foggy/','yolov5/datasets/cityscapes/images/dehazed_test/'],
        'yolo_train':['yolov5/datasets/cityscapes/images/train_foggy/','yolov5/datasets/cityscapes/images/dehazed_train/']}

if user_input not in path_:
    print("No valid choice entered, defaulting to yolo_inference")
    user_input='yolo_inference'


generate_dehazed_output(path_[user_input][0],path_[user_input][1])