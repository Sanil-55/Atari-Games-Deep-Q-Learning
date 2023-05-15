import numpy as np
import matplotlib.pyplot as plt
import sys

exp_replay = False
game_name = sys.argv[1]
if (len(sys.argv) > 2):
    if (sys.argv[2] == 'er'):
        exp_replay = True
    else:
        print ("Invalid Arguments")
        exit()
if exp_replay == True:
    all_rewards = np.load('Rewards_DQN_' + game_name + '_ExpReplay.npy')
else:
    all_rewards = np.load('Rewards_DQN_' + game_name + '.npy')

## Smoothening the Plot

start = 0
reward_plot_smooth = []
reward_plot_max = []
reward_plot_min = []
end = 50
x_axis = []

while (end < len(all_rewards)):
    cur = all_rewards[start : end]
    start = start + 10
    end = end + 10
    x_axis.append(end)
    reward_plot_smooth.append(1.0*sum(cur)/50.0)
    reward_plot_max.append(max(cur))
    reward_plot_min.append(min(cur))
    
plt.figure(figsize=(12,8)) 
plt.plot(x_axis, reward_plot_smooth)
plt.plot(x_axis, reward_plot_max)
plt.plot(x_axis, reward_plot_min)

if exp_replay == True:
    plt.title("DQN on " + game_name + " with Experience Replay")
else:
    plt.title("DQN on " + game_name)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend(['Average Performance of Set', 'Max Reward in Batch', 'Min Reward in Batch'], loc ="upper right")
plt.grid()

if exp_replay == True:
    plt.savefig('Rewards_DQN_' + game_name +'_ExpReplay.png')
else:
    plt.savefig('Rewards_DQN_' + game_name +'.png')
