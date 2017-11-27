""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import gym
import copy
import tensorflow as tf
import time

LR = 1e-4

# hyperparameters
num_channels1 = 16
num_channels2 = 32

#num_channels1 = 64
#num_channels2 = 128

num_H1 = 10
num_H2 = 100
eps0 = 1e-8


episode_limit = 10000



batch_size = 1 # every how many episodes to do a param update?
#learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False

# model initialization
D = 80  # input dimensionality: 80x80 grid

np.random.seed()
    



def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0:1] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  I2 = np.zeros( (1, D, D, 1) )
  I2[0,:] = I
  return I2



def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  #xrange in Python 2
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

#inputs
x0 = tf.placeholder(tf.float32, shape=[None, 80, 80,1])
#rewards
R = tf.placeholder(tf.float32, shape=[None])
#actions
A = tf.placeholder(tf.int32, shape=[None])

W1 = tf.Variable(tf.random_uniform([8,8,1,num_channels1], -1.0, 1.0)/np.sqrt(8*8))
W2 = tf.Variable(tf.random_uniform([8,8,num_channels1,num_channels2], -1.0, 1.0)/np.sqrt(8*8*num_channels1))
W3 = tf.Variable(tf.random_uniform([8,8,num_channels2,num_channels2], -1.0, 1.0)/np.sqrt(8*8*num_channels2))
W4 = tf.Variable(tf.random_uniform([5*5*num_channels2,num_H1], -1.0, 1.0)/np.sqrt(5*5*num_channels2))
W5 = tf.Variable(tf.random_uniform([num_H1, num_H2], -1.0, 1.0)/np.sqrt(num_H1))
W6 = tf.Variable(tf.random_uniform([num_H2, 2], -1.0, 1.0)/np.sqrt(num_H2))

C1 =  tf.nn.relu( tf.nn.conv2d(x0, W1, strides = [1,4,4,1], padding = "SAME")    )
C2 =   tf.nn.relu( tf.nn.conv2d(C1, W2, strides = [1,4,4,1], padding = "SAME")   )
C3 =  tf.nn.relu( tf.nn.conv2d(C2, W3, strides = [1,1,1,1], padding = "SAME")   )
#P1 = tf.nn.avg_pool(C2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

P1_flat =  tf.reshape(C3, [-1, 5*5*num_channels2])
H1 = tf.nn.relu( tf.matmul(P1_flat, W4 )    )
H2 =  tf.nn.relu(   tf.matmul(H1, W5)    )
u = tf.matmul(H2, W6) 

p = tf.nn.softmax(u)

cross_entropy_error = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=A , logits=u) 

objective_function = tf.constant(np.float32(1.0))*tf.multiply(R, cross_entropy_error   )
    
    
    
opt = tf.train.AdamOptimizer(learning_rate = LR, beta1 = 0.9, beta2 = .999)    
train_op = opt.minimize(objective_function)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)



env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,As,drs = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 0


time0 = time.time()
while (episode_number < episode_limit) :
  #print("check")
#  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros((1,D,D,1) )
  prev_x = cur_x
  
  # forward the policy network and sample an action from the returned probability
  x0000 = np.zeros( (2,D,D,1) )
 
  W1_np = session.run(W1)
  C2_np = session.run(C2, feed_dict={x0: x0000} )

  C3_np = session.run(C3, feed_dict={x0:x} )
  
  aprob = session.run(p, feed_dict={x0: x} ) 
  
  action = 2 if np.random.uniform() < aprob[0][0] else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation

  y = 0 if action == 2 else 1 # a "fake label"
  
  As.append(y)
    
  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    
  
  if ( done ): # an episode finished
   # print("check")
    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    N = np.float(len(drs))
    epr = np.vstack(drs)[:,0]
    A_np = np.int32( np.vstack(As)[:,0] )
    xs,As,drs = [],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    mean00 = np.mean(discounted_epr)
    std00 = np.std(discounted_epr)
    discounted_epr -= mean00
    discounted_epr /= std00
    
    
    session.run(train_op, feed_dict={x0: epx, A: A_np, R: discounted_epr } ) 
    
    time1 = time.time()
    current_time = time1-time0
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print (episode_number, reward_sum, running_reward, current_time )
            #last_reward = np.float(reward_sum_total)


    reward_sum = 0
    episode_number += 1
    observation = env.reset() # reset env
    prev_x = None



