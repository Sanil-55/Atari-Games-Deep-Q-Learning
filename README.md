## Term Paper Assignment - Neural Networks and Fuzzy Logic

### How to Run

- Clone the github project and navigate to DQN_Atari_NEAT Folder

- To run the Deep Q Network, dependencies are : NumPy, Torch, OpenAI's Gym, OpenCV and Seaborn

#### Without Experience Replay

(Game Name) can be Breakout, SpaceInvaders, Seaquest etc.
  
After installation of all above dependencies, to run DQN without Experience Replay : 
```sh
python DQN_main.py (Game Name) 
```
(Game Name) can be Breakout, SpaceInvaders, Seaquest etc.
  
The execution of above commands will result in formation of Rewards_DQN_(Game Name).npy and Q_DQN_(Game Name).npy.
  
To plot the results, run plotGraphRewards.py
```sh
python plotGraphRewards.py (Game Name)
```
To plot Maximum Action Values, run plotQValues.py
```sh
python plotQValues.py (Game Name)
```
The execution of the above commands will result in Rewards_DQN_(Game Name).png and Q_DQN_(Game Name).png which contain the graphs.
  
  
  
#### With Experience Replay

(Game Name) can be Breakout, SpaceInvaders, Seaquest etc.
```sh
python DQN_main.py (Game Name) er
```
(Game Name) can be Breakout, SpaceInvaders, Seaquest etc.
  
The execution of above commands will result in formation of Rewards_DQN_(Game Name)_ExpReplay.npy and Q_DQN_(Game Name)_ExpReplay.npy.
  
To plot the results, run plotGraphRewards.py
```sh
python plotGraphRewards.py (Game Name) er
```  
To plot Maximum Action Values, run plotQValues.py
```sh
python plotQValues.py (Game Name) er
```
The execution of the above commands will result in Rewards_DQN_<Game Name>_ExpReplay.png and Q_DQN_<Game Name>_ExpReplay.png which contain the graphs.
  
  
  ### To get Graphs of Preloaded weights, execute commands:

```sh
python plotGraphRewards.py Breakout er
python plotQValues.py Breakout er
```

  
- To run the NEAT Optimization for Hyperparameters, we need additional installation of NEAT-Python Library

Afer installation of NEAT-Python Library, to run the optimization :

```sh
python NEAT_Optimization/neat_gym.py <Game Name> 
```  


### File-wise Description

- **DQN_main.py** : Rewards and Average Maximum Action Values on Atari Games are calculated over episodes using OpenAI Gym Environment

- **plotGraphRewards.py** : Contains Python Script for creating Smoothened Average Rewards Graph from given .npy file```.

- **plotGraphQValues.py** : Contains Python Script for creating Smoothened Average Maximum Action Values Graph from given .npy file.

- **Performance/** : Folder containing all 16 plots as mentioned in assignment, training the DQN over 4 Atari Games, namely : Breakout, Space Invaders, QBert and Beam Rider.

- To calculate Average Maximum Action Values, we needed specific states starting from which RL Agent tries to maximize the reward. These sample states are stored in **Sample_Images/**

- **NEAT Optimization/** : Folder contains Implementation of Genetic Algorithm that uses NEAT-Python Framework to optimize structure and hyperparameters of Neural Network used for Training RL Agent on Space Invaders.
