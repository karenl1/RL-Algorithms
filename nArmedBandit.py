"""
Implementation of a policy-gradient based agent to solve the multi-armed bandit problem.
Following Tutorial 1 by Andrew Juliani.
"""

import tensorflow as tf 
import numpy as np 

# list out the bandits (currently bandit #4 with index #3 is set to most often provide a positive reward)
bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)
def pullBandit(bandit):
	# get a random number 
	result = np.random.randn()
	if result > bandit:
		# return a positive reward
		return 1
	else:
		# return a negative reward
		return -1

# establish the neural network
tf.reset_default_graph()
# feed-forward part of network (choosing which bandit)
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)
# establish the training procedure (feed the reward and chosen action into the network, then compute the loss and use it to update the network)
reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)
action_holder = tf.placeholder(shape = [1], dtype = tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
update = optimizer.minimize(loss)

# train the agent by taking actions in the environment and receiving rewards; update the network to more often choose actions that will yield the highest rewards over time
total_episodes = 1000  
total_reward = np.zeros(num_bandits)  # set scoreboard for bandits to 0
e = 0.1  # set the chance of taking a random action

init = tf.global_variables_initializer()

# launch the TensorFlow graph
with tf.Session() as sess:
	sess.run(init)
	i = 0
	while i < total_episodes:
		# choose either a random action or one from our network
		if np.random.rand() < e:
			action = np.random.randint(num_bandits)
		else:
			action = sess.run(chosen_action)
		# get reward from the chosen bandit
		reward = pullBandit(bandits[action])
		# update the network
		_, resp, ww = sess.run([update, responsible_weight, weights], feed_dict = {reward_holder:[reward], action_holder:[action]})
		# update our running tally of scores
		total_reward[action] += reward
		if i % 50 == 0:
			print "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward)
		i+=1

# print final results
print "The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising..."
if np.argmax(ww) == np.argmax(-np.array(bandits)):
	print "...and it was right!"
else:
	print "...and it was wrong!"