""" 
Implementation of the Q-Table algorithm in the OpenAI FrozenLake-v0 environment.
Following tutorial 0 by Arthur Juliani.
"""

import gym
import numpy as np 

# load the environment
env = gym.make('FrozenLake-v0')

# implement the Q-Table learning algorithm
# initalize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# set learning paramaters
print("Please set the learning parameters.")
lr = input("Learning rate, alpha (default = 0.85): ")
y = input("Discount factor, gamma (default = 0.99): ")
num_episodes = input("Number of episodes to run: ")
# create lists to contain total rewards and steps per episode
rList = []
jList = []
for i in range(num_episodes):
	# reset environment and get first new observation
	s = env.reset()
	rAll = 0
	d = False
	j = 0
	# implement Q-Table learning algorithm
	while j < 99:
		j+=1
		# choose an action by greedily (with noise) picking from Q-table
		a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
		s1, r, d, _ = env.step(a)
		# update Q-Table with new knowledge 
		Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
		rAll += r
		s = s1
		if d == True:
			break
	jList.append(j)
	rList.append(rAll)

# print output
print "Score over time: " + str(sum(rList)/num_episodes)
print "Final Q-Table Values"
print Q
