"""
Implementation of Q-Network Learning (one-layer neural network) to solve OpenAI's FrozenLake-v0 environment.
Following tutorial 0 by Arthur Juliani.
"""

import gym
import numpy as np 
import random
import tensorflow as tf 
import matplotlib.pyplot as plt 

# load the environment
env = gym.make('FrozenLake-v0')

# implement the network itself
tf.reset_default_graph()
# these lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape = [1,16], dtype = tf.float32)  # input for computation
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))  # weights for the neural network
Qout = tf.matmul(inputs1, W)  # matrix multiplication
predict = tf.argmax(Qout, 1)  # returns the index of the greatest entry in the vector
# obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape = [1,4], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextQ-Qout))  # adds up the squared differences for each element
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
updateModel = trainer.minimize(loss)

# train the network
init = tf.global_variables_initializer()
# set learning parameters
y = 0.99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
	sess.run(init)
	for i in range(num_episodes):
		# reset environment and get first new observation
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		# the Q-Network
		while j < 99:
			j+=1
			# choose an action greedily (with e chance of random action) from the Q-network
			a, allQ = sess.run([predict, Qout], feed_dict = {inputs1: np.identity(16)[s:s+1]})
			if np.random.rand(1) < e:
				a[0] = env.action_space.sample()
			# get new state and reward from environment
			s1, r, d, _ = env.step(a[0])
			# obtain the Q' values by feeding the new state through our network
			Q1 = sess.run(Qout, feed_dict = {inputs1: np.identity(16)[s1:s1+1]})
			# obtain maxQ' and set our target value for chosen action. 
			maxQ1 = np.max(Q1)
			targetQ = allQ
			targetQ[0, a[0]] = r + y*maxQ1
			# train our network using target and predicted Q-values
			_, W1 = sess.run([updateModel, W], feed_dict = {inputs1: np.identity(16)[s:s+1], nextQ: targetQ})
			rAll += r
			s = s1
			if d == True:
				# reduce chance of random action as we train the model
				e = 1./((i/50) + 10)
				break
		jList.append(j)
		rList.append(rAll)

# print output
print "Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%"