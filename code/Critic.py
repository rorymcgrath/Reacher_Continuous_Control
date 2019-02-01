import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
	'''The cirtic network used by the agent.'''

	def __init__(self,state_size,action_size,seed):
		'''Initlise and defined the model.

		Parameters
		----------
		state_size : int
			The Dimension of each state.

		action_size : int
			The dimension of each action

		seed : int
			The random seed used.
		'''
		super(Critic,self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size,148)
		self.fc2 = nn.Linear(148+action_size,148)
		self.fc3 = nn.Linear(148,1)

	def forward(self,state,action):
		'''Build the network that estimates Q values for each state.

		The action is added to the last hidden layer.
		#Why? it's mentioned in the DDQN paper but they don't reference why.
		Parameters
		----------
		state : array_like
			The current state.


		Returns
		-------
		Q_values : array_like
			The Q_values for each action given the current state.
		'''
		x = F.relu(self.fc1(state))
		x1 = torch.cat((x,action.float()),dim=1)
		return self.fc3(F.relu(self.fc2(x1)))


