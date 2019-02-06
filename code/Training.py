from collections import deque
from Agent import Agent
from unityagents import UnityEnvironment
import numpy as np
import torch
import pickle

NO_GRAPHICS = True
GPU_SERVER = False
MONITOR_INTERVAL = 10
TRAIN_MODE = True

env = UnityEnvironment(file_name='../Reacher_Linux_NoVis/Reacher.x86_64' if GPU_SERVER else '../Reacher.app', no_graphics=NO_GRAPHICS)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]

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

scores_window = deque(maxlen=100)
success_score = 30
scores = []
i_episode = 1
try:
	while True:
		env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]     
		state = env_info.vector_observations[0]                  
		score = 0
		done = False             
		while not done:
			action = agent.get_action(state,add_noise=TRAIN_MODE)[0]
			env_info = env.step(action)[brain_name]           
			next_state = env_info.vector_observations[0]         
			reward = env_info.rewards[0]                         
			done = env_info.local_done[0]                       
			agent.step(state,action,reward,next_state,done)
			score += reward                         
			state = next_state                             
		scores_window.append(score)
		scores.append(score)

		if i_episode%MONITOR_INTERVAL == 0:
			print('\rEpisode {} \tAverage Score for last {} episodes: {:.2f}'.format(i_episode, MONITOR_INTERVAL, np.mean(list(scores_window[i] for i in range(-1*MONITOR_INTERVAL,0)))))

		if i_episode%100 == 0:
			print('\rEpisode {} \tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

		if np.mean(scores_window) >= success_score:
			print('Environment solved in {:d} episodes. Average Score: {:.2f} Saving model parameters.'.format(i_episode-100,np.mean(scores_window)))
			torch.save(agent.actor_local.state_dict(), 'parameters/{}_actor_checkpoint.pth'.format(i_episode-100))
			torch.save(agent.critic_local.state_dict(), 'parameters/{}_critic_checkpoint.pth'.format(i_episode-100))
			success_score+=1
		i_episode+=1
except Exception as e:
	env.close()
	torch.save(agent.actor_local.state_dict(), 'parameters/recovery_actor_checkpoint.pth')
	torch.save(agent.critic_local.state_dict(), 'parameters/recovery_critic_checkpoint.pth')
	with open('recovery_scores.pkl','wb') as f:
		pickle.dump(scores,f)
	print(e)
	
env.close()
with open('scores.pkl','wb') as f:
	pickle.dump(scores,f)
