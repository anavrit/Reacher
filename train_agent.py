from unityagents import UnityEnvironment
from ddpg_agent import Agent

import torch
from collections import namedtuple, deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name="Reacher.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

def train_agent(agent, num_episodes = 1000, max_iter = 500, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    agent_scores = []
    best_mean_score = 30
    for episode in range(1, num_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations              # get the current state
        agent.reset()
        score = 0.
        individual_scores = np.zeros(num_agents)                # initialize the score
        for i in range(max_iter):
            actions = agent.act(states)                    # select an action
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                      # get the reward
            dones = env_info.local_done                     # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            score += np.mean(rewards)
            individual_scores += rewards                                # update the score
            states = next_states                             # roll over the state to next time step
            if np.any(dones):                                       # exit loop if episode finished
                break
        scores_deque.append(score)
        scores.append(score)
        agent_scores.append(individual_scores)
        print('\rEpisode {}\tReward: {:.2f}\tAverage Reward: {:.2f}'.format(episode, score, np.mean(scores_deque)), end="")
        if episode % print_every == 0:
            print('\rEpisode {}\tReward: {:.2f}\tAverage Reward: {:.2f}'.format(episode, score, np.mean(scores_deque)))
        if len(scores) >= 100:
            mean_score = np.mean(scores[-100:])
            if mean_score > best_mean_score:
                torch.save(agent.actor_local.state_dict(), 'actor_chkpoint.pth')
                torch.save(agent.critic_local.state_dict(), 'critic_chkpoint.pth')
                best_mean_score = mean_score
    return scores

agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)
scores = train_agent(agent)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

"""
hidden_layers = [256, 128, 64, 32]
filename = '../Resources/double_dqn_trained_weights_256x128x64x32_.pth'
agent = DoubleDQNAgent(state_size=state_size, action_size=action_size, seed=0, hidden_layers=hidden_layers)
scores = train_agent(agent, filename=filename, num_episodes = 1200)

# Saving plot of average scores over 100 episodes
fig = plt.figure(figsize=(8,6))
avg_100 = pd.Series([np.mean(scores[i-100:i]) for i in range(100, len(scores)+1)])
avg_100.index += 100
avg_100.plot()
plt.title('Tracking performance of RL agent')
plt.xlabel('Episode #')
plt.ylabel('Average score over 100 episodes')
plt.grid(color='gray', linestyle='--', linewidth=0.5, axis='y')
plt.axhline(y=13.0, linestyle='--', linewidth=1.0, color='red')
plt.savefig('../Resources/My_Trained_Agent.jpg', bbox_inches='tight')

# Game play
play = input('Would you like to play the trained agent (Y/N)? ')

while play in ["y", "Y"]:
    trained_weights = torch.load(filename)
    agent.qnetwork_local.load_state_dict(trained_weights)
    agent.qnetwork_target.load_state_dict(trained_weights)
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.step(state, action, reward, next_state, done)
        score += reward
        state = next_state
        if done:
            break
    print('Score =', score)
    play = input('Play again (Y/N)? ')
    if play not in ["y", "Y"]:
        break
"""

env.close()
