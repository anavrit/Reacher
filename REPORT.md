[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project #1: REPORT

For this project, I trained an agent on 20 identical agents in the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment, each with its own copy of the environment. The environment was solved using  using the starter code provided by Udacity.

![Trained Agent][image1]

### Learning Algorithm

The Deep Deterministic Policy Gradients (DDPG) algorithm is an off-policy, model-free policy gradient model inspired from the seminal paper entitled - CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING (Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver & Daan Wierstra, ICML, 2016.). Two neural networks - one for the actor, and the other for the critic - are trained to solve the Unity Reacher Environment. The actor and critic neural networks takes in 33-dimensional input (state size) from the Unity Reacher Environment. The actor network outputs 4 continuous (action) values between -1 and 1 corresponding to the torque applicable to the two joints (for each agent). Whereas the critic network outputs 1 value corresponding to the Q value for the input state and the action taken.

**Network Architecture**

ACTOR:

`Actor(
  (fc1): Linear(in_features=33, out_features=400, bias=True)
  (fc2): Linear(in_features=400, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=4, bias=True)
)`

The trained weights are available in `/Resources/checkpoint_actor.pth`.

CRITIC:

`Critic(
  (fcs1): Linear(in_features=33, out_features=400, bias=True)
  (fc2): Linear(in_features=404, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=1, bias=True)
)`

The trained weights are available in `/Resources/checkpoint_critic.pth`.

**Hyper-parameters**

After testing hyperparameters for batch size and learning rate, the following set of hyperparameters were used for all architectures of deep neural networks, including the selected network:

BUFFER_SIZE = int(1e5)  <br>
BATCH_SIZE = 128        <br>
GAMMA = 0.99            <br>
TAU = 1e-3              <br>
LR_ACTOR = 1e-4         <br>
LR_CRITIC = 1e-4        <br>
WEIGHT_DECAY = 0.       <br>

### Plot of Rewards

![Reacher Environment Solved](/Resources/Reacher_Environment_Solved.jpg)

The environment was solved in **22** episodes!	Average Reward: 36.62 <br>

### Ideas for Future Work

A number of algorithms have the potential to improve the performance of DDPG for the Reacher environment with 20 agents. A couple of key ideas for future work are:

1. [Trust Region Policy Optimization](https://arxiv.org/pdf/1604.06778.pdf) (TRPO) and Truncated Natural Policy Gradient (TNPG) should achieve better performance than DDPG but could be even further improved. <br><br>
2. [D4PG](https://openreview.net/pdf?id=SyZipzbCb) or the Distributional Deterministic Deep Policy Gradient algorithm, has been shown to achieve state of the art performance on a number of challenging continuous control problems.
