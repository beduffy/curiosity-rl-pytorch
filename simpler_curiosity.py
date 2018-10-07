import sys
import argparse
import gym
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.distributions import Categorical

from envs import create_atari_env

# Hyper-parameters
# input_size = 1
# output_size = 1
# num_epochs = 60
learning_rate = 0.01

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--env-name', default='Breakout-v0',
                    help='environment to train on (default: PongDeterministic-v4)')
args = parser.parse_args()



# env = gym.make('CartPole-v0').unwrapped
# env = gym.make('Breakout-v0')
env = create_atari_env(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps.item()

# resize = T.Compose([T.ToPILImage(),
#                     T.Resize(40, interpolation=Image.CUBIC),
#                     T.ToTensor()])

def select_action(state):
    # state = torch.from_numpy(state).float().unsqueeze(0)
    # state =
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob.cuda() * reward.cuda())
    policy_optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    policy_optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

class PolicyConv(nn.Module):
    def __init__(self, input_channels=1, num_actions=4):
        super(PolicyConv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.head = nn.Linear(128, num_actions)
        # self.head = nn.Linear(2048, 448)
        # self.head2 = nn.Linear(448, 2)
        # self.head2 = nn.Linear(448, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.head(x.view(x.size(0), -1)))
        action_scores = self.head(x.view(x.size(0), -1))
        # action_scores = self.head2(x)

        return F.softmax(action_scores, dim=1)

class ConvFeatureExtract(nn.Module):
    def __init__(self, num_input_channel=1):
        super(ConvFeatureExtract, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channel, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # self.head = nn.Linear(448, 40 * 80 * 3) # todo change to dim 20 or so and embed everything
        # self.head = nn.Linear(448, 128)
        self.head = nn.Linear(2048, 512)
        # self.head2 = nn.Linear(512, 128) # todo maybe?

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DynamicsModel(nn.Module):
    def __init__(self, encoded_state_size):
        super(DynamicsModel, self).__init__()
        # self.conv1 = nn.Conv2d(num_input_channel, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.head = nn.Linear(448, 40 * 80 * 3) # todo change to dim 20 or so and embed everything
        # self.head = nn.Linear(448, 128)
        self.state_head = nn.Linear(encoded_state_size + 1, encoded_state_size)
        # self.action_head = nn.Linear(1, state_size) # todo maybe one hot encode!!!!!!
        # self.head2 = nn.Linear(512, 128) # todo maybe?

    def forward(self, state, action):
        # import pdb; pdb.set_trace()
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        action = torch.Tensor([action]).unsqueeze(0).to(device)
        next_state_pred = self.state_head(torch.cat([state, action], 1))
        return next_state_pred


# feature_extractor_model = PolicyConv()
feature_extractor_model = ConvFeatureExtract().to(device)
dynamics_model = DynamicsModel(encoded_state_size=512).to(device)
policy = PolicyConv().to(device) # todo could try feature_extractor_model instead

criterion = nn.MSELoss()
dynamics_optimizer = torch.optim.SGD(dynamics_model.parameters(), lr=learning_rate)
policy_optimizer = optim.Adam(policy.parameters(), lr=1e-2)

stored_mses = []
episode_lengths = []

state = env.reset()

print(env.action_space)

# how much to pretrain
# inner_idx_anneal_stategy = lambda outer_idx: 1000 if outer_idx < 25 and episode == 0 else 10
inner_idx_anneal_stategy = lambda outer_idx: 1000 if outer_idx < 5 and episode == 0 else 10

for episode in range(100):
    print('Episode Number: {}'.format(episode))
    outer_idx = 0
    while True:
    # for outer_idx in range(2000):
        env.render()
        last_state = state
        # action = env.action_space.sample() # your agent here (this takes random actions)
        # import pdb;pdb.set_trace()
        # todo check processed state still has ball. Maybe add colour and more res

        processed_state = torch.from_numpy(state).unsqueeze(0).to(device)
        action = select_action(processed_state)
        state, reward, done, info = env.step(action)

        last_state_encoded = feature_extractor_model(torch.from_numpy(last_state).unsqueeze(0).to(device)).detach() # random features
        state_encoded = feature_extractor_model(torch.from_numpy(state).unsqueeze(0).to(device)).detach()

        final_inner_index = 0
        loss_aggregate = torch.Tensor([0])
        for inner_idx in range(inner_idx_anneal_stategy(outer_idx)):

            state_encoded_pred = dynamics_model(last_state_encoded.to(device), action)
            loss = criterion(state_encoded_pred, state_encoded)
            dynamics_optimizer.zero_grad()
            loss.backward()
            dynamics_optimizer.step()

            # loss_values.append(loss)
            loss_aggregate += loss.cpu()
            loss_value = loss.cpu().data.numpy() # todo should we average all the losses?

            if inner_idx % 100 == 0:
                print('{}: loss: {}'.format(inner_idx, loss_value))

            if loss < 0.0001:
                break
        final_inner_index = inner_idx
        if inner_idx_anneal_stategy(outer_idx) <= 100:
            # loss_avg = loss_values.mean()
            # loss_avg = loss_aggregate / inner_idx_anneal_stategy(outer_idx)
            loss_avg = loss_aggregate / (final_inner_index + 1)# + reward
            stored_mses.append(loss_avg)
            policy.rewards.append(loss_avg)

            if done:
                print('Episode over at: {}'.format(outer_idx))
                episode_lengths.append(outer_idx)
                finish_episode()
                state = env.reset()
                break
        outer_idx += 1


plt.plot(range(len(episode_lengths)), episode_lengths)
plt.title('Episode Lengths')
plt.show()

plt.plot(range(len(stored_mses)), stored_mses)
plt.title('Mean Square Error')
plt.show()
