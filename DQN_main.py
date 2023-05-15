import numpy as np
import torch
import time
from torch import nn
import gym
import cv2
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
import seaborn as sns
import sys

exp_replay = False
game_name = sys.argv[1] + "-v0"
if (len(sys.argv) > 2):
    if (sys.argv[2] == 'er'):
        exp_replay = True
    else:
        print ("Invalid Arguments")
        exit()

"""Building Environment for Atari Games using OpenAI's Gym Framework"""

env = gym.make(game_name)
game_name = sys.argv[1]
print('Action space:', env.action_space)
print('Observation space:', env.observation_space)

"""Following block defines Utility functions that help in preprocessing of observations from the environment."""

def filter_obs(obs, resize_shape=(84, 110), crop_shape=None):
    ''' filter_obs() is used to resize Images from Atari's gameplay to a uniform
    size of (84,110), this is done to reduce the computational cost 
    
    returns Resized Image in numpy array format 
    '''
    assert(type(obs) == np.ndarray), "The observation must be a numpy array!"
    assert(len(obs.shape) == 3), "The observation must be a 3D array!"

    obs = cv2.resize(obs, resize_shape, interpolation=cv2.INTER_LINEAR)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs / 255.
    if crop_shape:
        crop_x_margin = (resize_shape[1] - crop_shape[1]) // 2
        crop_y_margin = (resize_shape[0] - crop_shape[0]) // 2
        
        x_start, x_end = crop_x_margin, resize_shape[1] - crop_x_margin
        y_start, y_end = crop_y_margin, resize_shape[0] - crop_y_margin
        
        obs = obs[x_start:x_end, y_start:y_end]
    
    return obs

def get_stacked_obs(obs, prev_frames):
    ''' get_stacked_obs() is used to stack a few frames together to understand
    the current direction of movement of blocks and environment
    
    returns numpy array containing N_FRAMES frames of an observation'''

    if not prev_frames:
        prev_frames = [obs] * (N_FRAMES - 1)
        
    prev_frames.append(obs)
    stacked_frames = np.stack(prev_frames)
    prev_frames = prev_frames[-(N_FRAMES-1):]
    
    return stacked_frames, prev_frames
    
N_FRAMES = 4

def preprocess_obs(obs, prev_frames): 
    ''' utilizes above defined functions to complete preprocessing of images.

    returns N_FRAMES number of resized frames stacked together'''

    filtered_obs = filter_obs(obs)
    stacked_obs, prev_frames = get_stacked_obs(filtered_obs, prev_frames)
    return stacked_obs, prev_frames

def format_reward(reward):
    ''' Tries to normalize and bring uniformity to the reward'''
    
    if reward > 0:
        return 1
    elif reward < 0:
        return -1
    return 0

"""Following block defines Network of Deep Q Learning"""

class DQN(nn.Module):
    def __init__(self, n_acts):
        ''' Initializes Convolutional Neural Network Layers with similar parameters as
        that of Research Paper '''

        super(DQN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(N_FRAMES, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(32 * 12 * 9, 256),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(256, n_acts))
        
    def forward(self, obs):
        ''' Constructs forward pass through the Neural Network Layers initialized
        Neural Network acts as a function approximator for the Q - Function
        
        returns Q_values obtained for all of the actions possible'''

        q_values = self.layer1(obs)
        q_values = self.layer2(q_values)
        
        # 2015 model: (32, 8x8, 4), (64, 4x4, 2), (64, 3x3, 1), (512)
        q_values = q_values.view(-1, 32 * 12 * 9)
        q_values = self.layer3(q_values)
        q_values = self.layer4(q_values)
        
        return q_values
    
    def train_on_batch(self, optimizer, obs, acts, rewards, next_obs, terminals, gamma=0.99):
        ''' Trains a batch of set of observations, taking into account the discounted reward,
        and calculates the loss according to equations provided '''

        next_q_values = self.forward(next_obs)
        max_next_q_values = torch.max(next_q_values, dim=1)[0].detach()
        
        terminal_mods = 1 - terminals
        actual_qs = rewards + terminal_mods * gamma * max_next_q_values
            
        pred_qs = self.forward(obs)
        pred_qs = pred_qs.gather(index=acts.view(-1, 1), dim=1).view(-1)
        
        loss = torch.mean((actual_qs - pred_qs) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class Batch():
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        
    def add_to_batch(self, step_data):
        self.data.append(step_data)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]
            
    def getBatch(self, n):
        n = min(n, len(self.data))
        indices = [n-1]*n
        samples = np.asarray(self.data)[indices]
        
        state_data = torch.tensor(np.stack(samples[:, 0])).float()
        act_data = torch.tensor(np.stack(samples[:, 1])).long()
        reward_data = torch.tensor(np.stack(samples[:, 2])).float()
        next_state_data = torch.tensor(np.stack(samples[:, 3])).float()
        terminal_data = torch.tensor(np.stack(samples[:, 4])).float()
        
        return state_data, act_data, reward_data, next_state_data, terminal_data

n_episodes = 3000
max_steps = 1000
n_acts = env.action_space.n
train_batch_size = 32
learning_rate = 1e-3
print_freq = 10
update_freq = 4
frame_skip = 3
n_anneal_steps = 1e5
epsilon = lambda step: np.clip(1 - 0.9 * (step/n_anneal_steps), 0.1, 1)

## Initializing Model with Defined Class containing Structure for Neural Network
model = DQN(n_acts=env.action_space.n)

## Preparing batch for training
batch_set = Batch(train_batch_size)

## RMSProp Optimizer used similar to one used in Research Paper
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-6)

## all_rewards keeps a track of rewards obtained during each episode, helps in plotting values
all_rewards = []

## q_values_normalized keeps a track of stability of choice of Q_value for a particular action 
q_values_normalized = []

## global_step used in Episilon-Greedy strategy
global_step = 0

## Keeping track of Normalized Maximum Action Values obtained 
q_rewards = []

if exp_replay == False:
    for episode in range(n_episodes):
        ## Enacting an episode of training

        ## Initializing a new enivironment each episode and preprocessing observations
        prev_frames = []
        obs, prev_frames = preprocess_obs(env.reset(), prev_frames)
        episode_reward = 0
        step = 0


        while step < max_steps:
            ## Enacting a step taken by Agent
          
            obs_tensor = torch.tensor([obs]).float()
            q_values = model(obs_tensor)[0]
            q_values = q_values.cpu().detach().numpy()

            ## Deciding action based on Episilon - Greedy Strategy, balancing between exploration and exploitation
            if np.random.rand() < epsilon(global_step):
                act = np.random.choice(range(n_acts))
            else:    
                act = np.argmax(q_values)
            
            ## Implementing action chosen using Episilon - Greedy Strategy
            cumulative_reward = 0
            for _ in range(frame_skip):

                ## Obtaining next set of observations
                next_obs, reward, done, _ = env.step(act)
                cumulative_reward += reward
                if done or step >= max_steps:
                    break
            
            episode_reward += cumulative_reward
            reward = format_reward(cumulative_reward)

            ## Preprocessing next set of observations
            next_obs, prev_frames = preprocess_obs(next_obs, prev_frames)
            batch_set.add_to_batch([obs, act, reward, next_obs, int(done)])
            obs = next_obs
            
            ## Updating Weights at a defined frequency using defined function train_on_batch()
            if global_step % update_freq == 0:
                obs_data, act_data, reward_data, next_obs_data, terminal_data = batch_set.getBatch(train_batch_size)
                model.train_on_batch(optimizer, obs_data, act_data, reward_data, next_obs_data, terminal_data)
            
            step += 1
            global_step += 1
            
            if done:
                break

        ## Appending Reward for a particular episode to storage        
        all_rewards.append(episode_reward)
            
        
        '''Calculating Maximum Action Value predicted for a particular state, here a
        sample of 4 images has been obtained from the game '''

        prev_frames = [0, 0, 0]
        env.reset()

        ## Preprocessing Sample Images, same as above
        low_game_name = game_name.lower()
        obs_arr = [np.array(mpimg.imread("Sample_Images/img_" + low_game_name + "_4.png"))]
        for obs in obs_arr:
            
            filtered_obs = filter_obs(obs)
            prev_frames[0] = filter_obs(np.array(mpimg.imread("Sample_Images/img_" + low_game_name + "_1.png")))
            prev_frames[1] = filter_obs(np.array(mpimg.imread("Sample_Images/img_" + low_game_name + "_2.png")))
            prev_frames[2] = filter_obs(np.array(mpimg.imread("Sample_Images/img_" + low_game_name + "_3.png")))


            obs, prev_frames = preprocess_obs(obs, prev_frames)
            
            ## Keeping Track of Maximum, Mean and Standard Deviations for each of the steps taken
            step = 0
            max_q_value = []
            mean_q_value = []
            std_q_value = []
            
            ## Similar steps as taken above
            while step < max_steps:
                obs_tensor = torch.tensor([obs]).float()
                q_values = model(obs_tensor)[0]
                q_values = q_values.cpu().detach().numpy()
                
                if np.random.rand() < epsilon(global_step):
                    act = np.random.choice(range(n_acts))
                else:
                    act = np.argmax(q_values)

                cumulative_reward = 0
                for _ in range(frame_skip):
                    next_obs, reward, done, _ = env.step(act)
                    cumulative_reward += reward
                    if done or step >= max_steps:
                        break

                reward = format_reward(cumulative_reward)
                
                if done:
                    break
                
                step += 1
                
                ## Appending values obtained in a step to arrays defined
                max_q_value.append(q_values[act])
                mean_q_value.append(np.mean(q_values))
                std_q_value.append(np.std(q_values))

            ## Normalizing Maximum Action Value Pair, and appending to memory
            q_rewards.append((sum(max_q_value) - sum(mean_q_value))*1.0/sum(std_q_value))

        if episode % print_freq == 0:

            ## Printing Results obtained

            print('Episode #{} | Step #{} | Epsilon {:.2f} | Avg. Reward {:.2f} | Avg. Maximum Action Value : {:.2f}'.format(
                episode, global_step, epsilon(global_step), np.mean(all_rewards[-print_freq:]), np.mean(q_rewards[-print_freq:])))
            
            ## Saving Results obtained at fixed frequencies

            np.save("Rewards_DQN_" + game_name + ".npy", all_rewards)
            np.save("Q_DQN_" + game_name + ".npy", q_rewards)
    exit()   

"""Next Cell implements the procedure for Experience Replay Method"""

class ExperienceReplay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        
    def add_step(self, step_data):
        self.data.append(step_data)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]
            
    def sample(self, n):

        ## Preparing batch of randomized samples from stored data

        n = min(n, len(self.data))
        indices = np.random.choice(range(len(self.data)), n, replace=False)
        samples = np.asarray(self.data)[indices]
        
        ## Stacking them together, to provide a pipeline for training
        
        state_data = torch.tensor(np.stack(samples[:, 0])).float()
        act_data = torch.tensor(np.stack(samples[:, 1])).long()
        reward_data = torch.tensor(np.stack(samples[:, 2])).float()
        next_state_data = torch.tensor(np.stack(samples[:, 3])).float()
        terminal_data = torch.tensor(np.stack(samples[:, 4])).float()
        
        return state_data, act_data, reward_data, next_state_data, terminal_data

## Initializing the Scheme for Experience Replay Algorithm

er_capacity = 150000
er = ExperienceReplay(er_capacity)

"""Training Procedure for Deep Q Learning with Experience Replay"""

## Initializing Model with Defined Class containing Structure for Neural Network
model = DQN(n_acts=env.action_space.n)

## RMSProp Optimizer used similar to one used in Research Paper
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-6)

## all_rewards keeps a track of rewards obtained during each episode, helps in plotting values
all_rewards = []

## q_values_normalized keeps a track of stability of choice of Q_value for a particular action 
q_values_normalized = []

## global_step used in Episilon-Greedy strategy
global_step = 0

## Keeping track of Normalized Maximum Action Values obtained 
q_rewards = []

for episode in range(n_episodes):
    ## Enacting an episode of training

    ## Initializing a new enivironment each episode and preprocessing observations
    prev_frames = []
    obs, prev_frames = preprocess_obs(env.reset(), prev_frames)
    episode_reward = 0
    step = 0


    while step < max_steps:
        ## Enacting a step taken by Agent
      
        ## Deciding action based on Episilon - Greedy Strategy, balancing between exploration and exploitation
        if np.random.rand() < epsilon(global_step):
            act = np.random.choice(range(n_acts))
        else:
            obs_tensor = torch.tensor([obs]).float()
            q_values = model(obs_tensor)[0]
            q_values = q_values.cpu().detach().numpy()
            act = np.argmax(q_values)
        
        ## Implementing action chosen using Episilon - Greedy Strategy
        cumulative_reward = 0
        for _ in range(frame_skip):

            ## Obtaining next set of observations
            next_obs, reward, done, _ = env.step(act)
            cumulative_reward += reward
            if done or step >= max_steps:
                break
        
        episode_reward += cumulative_reward
        reward = format_reward(cumulative_reward)

        ## Preprocessing next set of observations
        next_obs, prev_frames = preprocess_obs(next_obs, prev_frames)

        ## Adding to Experience Replay Storage
        er.add_step([obs, act, reward, next_obs, int(done)])
        obs = next_obs
        
        ## Updating Weights at a defined frequency using defined function train_on_batch()
        if global_step % update_freq == 0:

            ## Retrieving Randomized Sample from Storage
            obs_data, act_data, reward_data, next_obs_data, terminal_data = er.sample(train_batch_size)
            model.train_on_batch(optimizer, obs_data, act_data, reward_data, next_obs_data, terminal_data)
        
        step += 1
        global_step += 1
        
        if done:
            break

    ## Appending Reward for a particular episode to storage        
    all_rewards.append(episode_reward)
        
 
    '''Calculating Maximum Action Value predicted for a particular state, here a
    sample of 4 images has been obtained from the game '''

    prev_frames = [0, 0, 0]
    env.reset()

    ## Preprocessing Sample Images, same as above
    low_game_name = game_name.lower()
    obs_arr = [np.array(mpimg.imread("Sample_Images/img_" + low_game_name + "_4.png"))]
    for obs in obs_arr:
        
        filtered_obs = filter_obs(obs)
        prev_frames[0] = filter_obs(np.array(mpimg.imread("Sample_Images/img_" + low_game_name + "_1.png")))
        prev_frames[1] = filter_obs(np.array(mpimg.imread("Sample_Images/img_" + low_game_name + "_2.png")))
        prev_frames[2] = filter_obs(np.array(mpimg.imread("Sample_Images/img_" + low_game_name + "_3.png")))

        obs, prev_frames = preprocess_obs(obs, prev_frames)
        
        ## Keeping Track of Maximum, Mean and Standard Deviations for each of the steps taken
        step = 0
        max_q_value = []
        mean_q_value = []
        std_q_value = []
        
        ## Similar steps as taken above
        while step < max_steps:
            obs_tensor = torch.tensor([obs]).float()
            q_values = model(obs_tensor)[0]
            q_values = q_values.cpu().detach().numpy()
            
            if np.random.rand() < epsilon(global_step):
                act = np.random.choice(range(n_acts))
            else:
                act = np.argmax(q_values)

            cumulative_reward = 0
            for _ in range(frame_skip):
                next_obs, reward, done, _ = env.step(act)
                cumulative_reward += reward
                if done or step >= max_steps:
                    break

            reward = format_reward(cumulative_reward)
            
            if done:
                break
            
            step += 1
            
            ## Appending values obtained in a step to arrays defined
            max_q_value.append(q_values[act])
            mean_q_value.append(np.mean(q_values))
            std_q_value.append(np.std(q_values))

        ## Normalizing Maximum Action Value Pair, and appending to memory
        q_rewards.append((sum(max_q_value) - sum(mean_q_value))*1.0/sum(std_q_value))

    if episode % print_freq == 0:

        ## Printing Results obtained

        print('Episode #{} | Step #{} | Epsilon {:.2f} | Avg. Reward {:.2f} | Avg. Maximum Action Value : {:.2f}'.format(
            episode, global_step, epsilon(global_step), np.mean(all_rewards[-print_freq:]), np.mean(q_rewards[-print_freq:])))
        
        ## Saving Results obtained at fixed frequencies

        np.save('Rewards_DQN_' + game_name + '_ExpReplay.npy', all_rewards)
        np.save('Q_DQN_' + game_name + '_ExpReplay.npy', q_rewards)
