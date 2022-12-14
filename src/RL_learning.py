import gymnasium
import sys
import os
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


from dqn import DQN

from collections import namedtuple, deque
from itertools import count

import math
import random
import numpy as np

from vgg_features import extract_vgg_features
import argparse
import _init_

# sys.path.append(os.path.abspath("/Users/iamariyap/Desktop/sem3/PredictiveML/RL_Project/code/PMLProject/src/yolov5"))
sys.path.append(os.path.abspath("./yolov5"))


parser = argparse.ArgumentParser(description='RL learning module')
parser.add_argument('--data', required=False,default=1,
                    help='The number associated with the data file ')
args = parser.parse_args()
data_file_no=args.data
print("file selected: ",data_file_no)
#Dehaze Agenet environment is created
env = gymnasium.make('env/DehazeAgent-v0', render_mode='human',data_file_no=data_file_no).unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        random_trajs=random.sample(self.memory, batch_size)
        return random_trajs

    def __len__(self):
        return len(self.memory)
    
    

env.reset()


# Params for RL training
BATCH_SIZE = 8
# Discount rate
GAMMA = 0.999
# Epsilon related parameters
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)

files=os.listdir()
if 'dqn_policy_net.pt' in files :
    policy_net.load_state_dict(torch.load("dqn_policy_net.pt")) # load the previous weigths to continue

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)
        
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # print("non final next states shape ", non_final_next_states.shape)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
num_episodes = 25
for i_episode in range(num_episodes):
    # Initialize the environment and state
    observation, info = env.reset()
    state = observation["image"]
    features = extract_vgg_features(state)
    print("Episode : ",i_episode)
    for t in count():
        # Select and perform an action

        
        # print("Size of the features " ,  features.shape)
        action = select_action(features)
        observation, reward, done, _, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = observation["image"]
            next_features = extract_vgg_features(next_state)
        else:
            next_state = None
            next_features = None

        # if next_state is not None:
        # Store the transition in memory
        
        memory.push(features, action, next_features, reward)

        # Move to the next state
        state = next_state
        next_features = features
        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break

        # Update the target network, copying all weights and biases in DQN
        if t % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    if i_episode % 10:
        torch.save(target_net.state_dict(), "./dqn_target_net.pt")
        torch.save(policy_net.state_dict(), "./dqn_policy_net.pt")

torch.save(target_net.state_dict(), "./dqn_target_net.pt")
torch.save(policy_net.state_dict(), "./dqn_policy_net.pt")

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

