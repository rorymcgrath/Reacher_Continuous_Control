import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
	'''The Actor Q network used by the agent.'''

	def __init__(self,state_size,action_size,seed):
		'''Initlise and defined the model.

		Parameters
		----------
		state_size : int
			The Dimension of each state.

		action_size : int
			The number of actions action.

		seed : int
			The random seed used.
		'''
		super(Actor,self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size,148)
		self.fc2 = nn.Linear(148,148)
		self.fc3 = nn.Linear(148,action_size)

	def forward(self,state):
		'''Build the network that estimates Q values for each state.

		Parameters
		----------
		state : array_like
			The current state.


		Returns
		-------
		Q_values : array_like
			The Q_values for each action given the current state.
		'''
		#using tanh to bound the return values to acceptiable range.
		#how exactly does this work? it is bound from -1 to 1
		return F.tanh(self.fc3(F.relu(self.fc2(F.relu(self.fc1(state))))))
