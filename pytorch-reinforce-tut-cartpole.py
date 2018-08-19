"""
Reinforce code borrowed from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
"""

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


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0').unwrapped
env.seed(args.seed)
torch.manual_seed(args.seed)
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# This is based on the code from gym.
screen_width = 600


def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

class PolicyFF(nn.Module):
    def __init__(self, num_input_params, num_output_params=2):
        super(PolicyFF, self).__init__()
        self.affine1 = nn.Linear(num_input_params, 128)
        self.affine2 = nn.Linear(128, num_output_params)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class PolicyConv(nn.Module):
    def __init__(self):
        super(PolicyConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        action_scores = self.head(x.view(x.size(0), -1))

        return F.softmax(action_scores, dim=1)

class SurprisalModelFF(nn.Module):
    def __init__(self):
        super(SurprisalModelFF, self).__init__()
        self.affine1 = nn.Linear(5, 128)
        self.affine2 = nn.Linear(128, 64)
        self.affine3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = self.affine2(x)
        x_t_plus_1 = self.affine3(x)
        return x_t_plus_1

class SurprisalModelConv(nn.Module):
    def __init__(self):
        super(SurprisalModelConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 40 * 80 * 3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


# normal cartpole
# policy = Policy(4, 2)

# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()
# sys.exit()
# env.reset()
# state = get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy()
# print(state.shape)
policy = PolicyConv().cuda()
# policy = PolicyFF()


# surprise_model = SurprisalModelFF()
surprise_model = SurprisalModelConv().cuda()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
surprise_optimizer = optim.Adam(surprise_model.parameters(), lr=1e-4)
eps = np.finfo(np.float32).eps.item()

stored_mses = []

def pretrain_surprise_model():
    print('Pre training surprise model')
    for _ in range(1):
        state = prev_state = env.reset()
        prev_state = state = get_screen()
        # todo find out if resetting state changes initial
        for t in range(2500):  # Don't infinite loop while learning

            # prev_state = state
            # prev_state = state = get_screen()
            # action = select_action(state)
            action = 0
            # don't run the action loop, just try to predict the first image
            # state, reward, done, _ = env.step(action)

            reward = get_mse_reward_prediction_error(prev_state, action, state)


        print('First MSE: {}. Last: {}. Average: {}'.format(stored_mses[0], stored_mses[-1], sum(stored_mses) / len(stored_mses)))
        # plt.plot(stored_mses)
        # plt.ylabel('some numbers')
        # plt.show()

        # output = surprise_model(state)
        # plt.figure()
        # plt.title('Output of generative model')
        # plt.imshow(output.view(-1, 3, 80, 40).cpu().squeeze(0).permute(1, 2, 0).detach().numpy(),
        #            interpolation='none')
        # plt.show()
        # sys.exit()

# reward based on curiosity
def get_mse_reward_prediction_error_from_numpy(x_t, a_t, x_t_plus_1):
    # no need because 4 features
    # x_t_emb = embel_model(x_t)
    # a_t_emb = embel_model(x_t_plus_1)

    x_t_plus_1 = torch.from_numpy(x_t_plus_1).float().unsqueeze(0)

    # learned dynamics model f
    x_t = torch.from_numpy(x_t).float()#.unsqueeze(0)
    a_t = torch.from_numpy(np.array(a_t)).float().unsqueeze(0)#.unsqueeze(0)

    input_to_model = torch.cat([x_t, a_t]).unsqueeze(0)
    x_t_plus_1_pred = surprise_model(input_to_model)
    mse = torch.sum((x_t_plus_1_pred - x_t_plus_1) ** 2)

    # train right here boom. But maybe move outside
    surprise_optimizer.zero_grad()
    mse.backward()
    surprise_optimizer.step()
    # print(x_t_plus_1_pred)
    # print(x_t_plus_1)
    # print(mse.item())
    stored_mses.append(mse.item())
    return mse

# reward based on curiosity
def get_mse_reward_prediction_error(x_t, a_t, x_t_plus_1):
    a_t = torch.from_numpy(np.array(a_t)).float().unsqueeze(0).to(device)#.unsqueeze(0)

    # input_to_model = torch.cat([x_t, a_t]).unsqueeze(0)
    input_to_model = x_t
    x_t_plus_1_pred = surprise_model(input_to_model)
    mse = torch.sum((x_t_plus_1_pred - x_t_plus_1.view(1, -1)) ** 2)

    # train right here boom. But maybe move outside
    surprise_optimizer.zero_grad()
    mse.backward()
    surprise_optimizer.step()

    # print(x_t_plus_1_pred)
    # print(x_t_plus_1)
    # print(mse.item())
    stored_mses.append(mse.item())
    return mse

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
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    pretrain_surprise_model()
    stored_mses = []
    episode_lengths = []
    # sys.exit()
    running_reward = 10
    for i_episode in range(500):
        state = prev_state = env.reset()
        for t in range(1000):  # Don't infinite loop while learning
            state = get_screen()
            prev_state = state
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            state = get_screen()
            # if args.render:
            # env.render()

            # reward = get_mse_reward_prediction_error(prev_state, action, state)
            #
            # todo
            # Reward normalization. Since the reward function is non-stationary, it is useful to normalize
            # the scale of the rewards so that the value function can learn quickly. We did this by dividing
            # the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
            # reward_norm = reward / est_stddev_of_sum_of_disc_reward
            # reward_norm = (reward / (1 if len(stored_mses) < 1 else np.std([mse.item() for mse in stored_mses])))
            # reward_norm = (reward / (1 if len(stored_mses) < 500000 else np.std([mse.item() for mse in stored_mses])))
            # above made it worse?

            # reward_norm
            reward_norm = reward # todo change
            stored_mses.append(reward_norm)
            policy.rewards.append(reward_norm)
            # stored_mses.append(reward)
            # policy.rewards.append(reward)
            if done:
                print(t)
                episode_lengths.append(t)
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            # print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            #     i_episode, t, running_reward))
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                  i_episode, t, sum(episode_lengths) / len(episode_lengths)))

            plt.figure(2)
            plt.clf()
            # durations_t = torch.tensor(episode_durations, dtype=torch.float)
            plt.title('Training...')
            plt.xlabel('Episode')
            plt.ylabel('Duration')
            plt.plot(stored_mses)
            # Take 100 episode averages and plot them too
            # if len(durations_t) >= 100:
            #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            #     means = torch.cat((torch.zeros(99), means))
            #     plt.plot(means.numpy())

            plt.pause(0.001)  # pause a bit so that plots are updated
            # plt.plot(stored_mses)
            # plt.ylabel('some numbers')
            # plt.show()
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
