"""
Implementation of a policy-gradient based agent that can solve OpenAI's CartPole-v0 environment.
The neural network takes observations, passes them through a single hidden layer, then outputs the probability of choosing a left/right movement.
Following Tutorial 2 by Arthur Juliani.
"""

import numpy as np 
import cPickle as cPickle
import tensorflow as tf 
import matplotlib.pyplot as plt 
import math
import gym 

# load the CartPole-v0 environment
env = gym.make('CartPole-v0')

# hyper parameters for the neural network
H = 10  # number of hidden layer neurons
batch_size = 5  # every how many episodes to do a parameter update?
learning_rate = 1e-2  # can vary this to train faster or more stably
gamma = 0.99  # discount factor for reward
D = 4  # dimensions for the input

tf.reset_default_graph()

# define the network as it goes from taking an observation of the environment to giving a probabilty of choosing the action of moving left or right
observations = tf.placeholder(tf.float32, [None,D], name = "input_x")
W1 = tf.get_variable("W1", shape = [D, H], initializer = tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape = [H, 1], initializer = tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)
# define parts of the network needed for learning a good policy
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name = "input_y")
advantages = tf.placeholder(tf.float32, name = "reward_signal")

# the loss function that sends the weights in the direction of making actions that give good advantage (reward over time) more likely, and actions that didn't less likely
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

# once we have collected a series of gradients from multiple episodes, we apply them
# we don't just apply gradients after every episodes in order to account for noise in the reward signall
adam = tf.train.AdamOptimizer(learning_rate = learning_rate)  # our optimizer
W1Grad = tf.placeholder(tf.float32, name = "batch_grad1")  # placeholders to send the final gradients through when we update
W2Grad = tf.placeholder(tf.float32, name = "batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

# weigh the rewards that the agent receives
# for the CartPole environment, we want actions that keep the pole in the air for a long time to have a large reward, and actions that contributed to the pole falling to have a decreased or negative reward
# weigh the rewards from the end of the episode negatively (likely contributed to pole falling) and weigh earlier actions more positively (weren't responsible for the pole falling)
def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted rewards """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

# run the neural network in the CartPole-v0 environment
xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()
# launch the graph
with tf.Session() as sess:
	rendering = False
	sess.run(init)
	observation = env.reset()  # obtain an initial observation of the environment
	# reset the gradient placeholder; will collect gradients in gradBuffer until we are ready to update our policy network
	gradBuffer = sess.run(tvars)
	for ix, grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0  # set all elements in gradBuffer equal to 0

	while episode_number <= total_episodes:
		# rendering the environment slows things down, so only look at it once agent is doing good job
		if reward_sum/batch_size > 100 or rendering == True:
			env.render()
			rendering = True
		# make sure the observation is in a shape the network can handle
		x = np.reshape(observation, [1, D])
		# run the policy network and get an action to take
		tfprob = sess.run(probability, feed_dict = {observations: x})
		action = 1 if np.random.uniform() < tfprob else 0
		xs.append(x)  # observation
		y = 1 if action == 0 else 0  # a "fake label" (holds opposite value as xs)
		ys.append(y)
		# step the environment and get new measurements
		observation, reward, done, info = env.step(action)
		reward_sum += reward
		# record reward (has to be done after we call step() to get reward for previous action)
		drs.append(reward)

		if done:
			episode_number += 1
			# stack together all inputs, hidden states, action gradients, and rewards for this episode
			epx = np.vstack(xs)
			epy = np.vstack(ys)
			epr = np.vstack(drs)
			tfp = tfps
			xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []  # reset array memory
			# compute the discounted reward backwards through time
			discounted_epr = discount_rewards(epr)
			# size the rewards to be unit normal (helps control the gradient estimator variance)
			discounted_epr -= np.mean(discounted_epr)
			discounted_epr /= np.std(discounted_epr)
			# get the gradient for this episode, and save it in the gradBuffer
			tGrad = sess.run(newGrads, feed_dict = {observations: epx, input_y: epy, advantages: discounted_epr})
			for ix, grad in enumerate(tGrad):
				gradBuffer[ix] += grad
			# if we have completed enough episodes, then update the policy network with our gradients
			if episode_number % batch_size == 0:
				sess.run(updateGrads, feed_dict = {W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
				for ix, grad in enumerate(gradBuffer):
					gradBuffer[ix] = grad * 0
				# give a summary of how well our network is doing for each batch of episodes
				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				print 'Average reward for episode %f. Total average reward %f.' % (reward_sum/batch_size, running_reward/batch_size)
				if reward_sum/batch_size > 200:
					print "Task solved in", episode_number, "episodes!"
					break
				reward_sum = 0
			observation = env.reset()

		print episode_number, "Episodes completed."