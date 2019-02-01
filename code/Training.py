from collections import deque
from Agent import Agent
from unityagents import UnityEnvironment
import numpy as np
import torch
import pickle

NO_GRAPHICS = True

env = UnityEnvironment(file_name='../Reacher.app', no_graphics=NO_GRAPHICS)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = Agent(state_size, action_size, seed=0)

env_info = env.reset(train_mode=True)[brain_name]     
state = env_info.vector_observations[0]                  
scores_window = deque(maxlen=10)
success_score = 30
scores = []
for i_episode in range(1500):
	score = 0
	done = False             
	while not done:
		action = agent.get_action(state)[0]
		env_info = env.step(action)[brain_name]           
		next_state = env_info.vector_observations[0]         
		reward = env_info.rewards[0]                         
		done = env_info.local_done[0]                       
		agent.step(state,action,reward,next_state,done)
		score += reward                         
		state = next_state                               
	scores_window.append(score)
	scores.append(score)

	if i_episode%10 == 0:
		print('\rEpisode {} \tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

	if np.mean(scores_window) >= success_score:
		print('Environment solved in {:d} episodes. Average Score: {:.2f} Saving model parameters.'.format(i_episode-10,np.mean(scores_window)))
		torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
		success_score+=1
env.close()

with open('scores.pkl','wb') as f:
	pickle.dump(scores,f)
