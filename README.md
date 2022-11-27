[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Continuous Control

### Introduction

For this project, I trained an agent on 20 identical agents in the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment, each with its own copy of the environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The barrier for solving the environment is to take into account the presence of many agents.  In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, rewards that each agent received are added up without discounting, to get a score for each agent.  This yields 20 potentially different scores.  We then take the average of these 20 scores.
- This yields an **average score** for each episode where the average is over all 20 agents.

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

### Getting Started

1. Create and activate a new environment with Python 3.6

  **Linux or Mac:**<br>
  `conda create --name drlnd python=3.6` <br>
  `conda activate drlnd`

  **Windows:**<br>
  `conda create --name drlnd python=3.6`<br>
  `activate drlnd`    

2. Install OpenAI gym in the environment:

  `pip install gym` <br>

3. Clone the following repository and install the additional dependencies:

  `git clone https://github.com/anavrit/Reacher.git`<br>
  `cd Reacher`<br>
  `pip install -r requirements.txt`

1. Download the environment that matches your operating system from one of the links below:

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

2. Move or copy the downloaded environment to the root directory of Reacher; and unzip the file to get `Reacher.app`.

### Instructions

A brief description of files in the `Python` folder: <br>
- `ddpg_agent.py`: defines a DDPG agent
- `model.py`: deep neural network model architecture
- `train_agent.py`: code used to train a DDPG agents

#### Train agent <br>

**Note:** For MacOS users you may have to enable firewall access to Reacher.app and give access to the app through Security & Privacy settings. Instructions are [here](https://support.apple.com/guide/mac-help/block-connections-to-your-mac-with-a-firewall-mh34041/mac).

1. Navigate to the Python directory

  `cd Python`

2. Train agent with the following command:

  `python train_agent.py`<br>

#### Resources <br>

The following key resources can be found in the `Resources` folder:

1. `checkpoint_actor.pth`: trained weights of the best network for the actor

2. `checkpoint_critic.pth`: trained weights of the best network for the critic

3. `Reacher_Environment_Solved.jpg`: graph tracking progress of the trained agent
