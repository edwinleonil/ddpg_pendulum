import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from Generic_ddpg import Agent

# Instantiate the Environment and Agent
env = gym.make('Pendulum-v0')
# env = gym.make('LunarLanderContinuous-v2')
env.seed(2)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=2)

# agent.actor_local.load_state_dict(torch.load('Pendulum_model.pth'))

# Train the Agent with DDPG
def ddpg(n_episodes=500, max_t=300, print_every=20):
    scores_deque = deque(maxlen=print_every)
    scores = []

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state) # predict action
            next_state, reward, done, _ = env.step(action) # execute action
            agent.step(state, action, reward, next_state, done) # save states into replay buffer
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        torch.save(agent.actor_local.state_dict(), 'Pendulum_model.pth')
        # torch.save(agent.actor_local.state_dict(), 'LunarLander_model.pth')
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= -150:
            print("Solved")
            break
            
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()