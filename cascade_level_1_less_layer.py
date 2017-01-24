#This code is going to build a CNN for cascaded level 1 
#This level is served as a rough estimation of the landmark


import tensorflow as tf
import numpy as np
from six.moves import xrange
import utils
import pickle
import time
import sys

################## Marco Definition ##########################

# DATA MARCO DEFINITION
# Pixel data type
IMAGE_DATA_TYPE = tf.float32
# Coordinate data type 
LABEL_DATA_TYPE = tf.float32
# Number of points to be detected on the face
N_POINTS = 68
# Image length & width
IMAGE_SIZE = 128
# 3 for color images, 1 for grayscale images
NUM_CHANNELS = 3

# CNN MARCO DEFINITION
# SEED for initializing CNN
# Set to None for random seed , Default : 66478
SEED = None
BATCH_SIZE = 500
NUM_EPOCHS = 250
LAMDA = 1e-5
STDDEV = 1
# Evaluation size
EVAL_BATCH_SIZE = 1108
EVAL_FREQUENCY =25
LEARNING_RATE = 0.001

##################### Get Data ###############################


train_files= []
train_files.append('./data/CNN_data/my_version_afw_train_128_128_color_mirror_CNN.pkl')
train_files.append('./data/CNN_data/my_version_helen_train_128_128_color_mirror_CNN.pkl')
train_files.append('./data/CNN_data/my_version_lfpw_train_128_128_color_mirror_CNN.pkl')
train_files.append('./data/CNN_data/my_version_ibug_train_128_128_color_mirror_CNN.pkl')

test_files = []
test_files.append('./data/CNN_data/my_version_helen_test_128_128_color_mirror_CNN.pkl')
test_files.append('./data/CNN_data/my_version_lfpw_test_128_128_color_mirror_CNN.pkl')

train_data = []
test_data = []

for train_file in train_files:
	train_data.extend(pickle.load(open(train_file,'rb')))

for test_file in test_files:
	test_data.extend(pickle.load(open(test_file,'rb')))

train_images=[]
test_images=[]
train_labels=[]
test_labels=[]

train_images[:] = [data.image for data in train_data]
test_images[:] = [data.image for data in test_data]
train_labels[:] = [data.landmarks for data in train_data]
test_labels[:] = [data.landmarks for data in test_data]

# train images shape (6566,128,128,3)
# test images shape (1108, 128, 128, 3)
# train label shape (6566, 136)
# test label shape(1108, 136)
train_images = np.asarray(train_images)
test_images = np.asarray(test_images)
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

# Change shape from (n, 68, 2) to (n,136)
train_labels = np.reshape(train_labels,(train_labels.shape[0],2*train_labels.shape[1]))
test_labels = np.reshape(test_labels,(test_labels.shape[0],2*test_labels.shape[1]))

train_size = train_images.shape[0]
test_size = test_images.shape[0]

assert train_labels.shape[1] == test_labels.shape[1],"Different number of landmarks between train and test"
assert test_labels.shape[1] == 2*N_POINTS,"Different number of landmarks between data and Marco definition, Please verify Marco Def"
assert train_images.shape[1] == test_images.shape[1] == IMAGE_SIZE, "Different image size between Marco def and Data, please verify"

print('Finished loading data')


#################### Build Graph #############################
# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.
train_images_node = tf.placeholder(
	IMAGE_DATA_TYPE,
	shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(
	LABEL_DATA_TYPE,shape=(BATCH_SIZE,N_POINTS*2))
test_images_node = tf.placeholder(
	IMAGE_DATA_TYPE,
	shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
test_labels_node = tf.placeholder(
	LABEL_DATA_TYPE,shape=(EVAL_BATCH_SIZE,N_POINTS*2))


# The variables below hold all the trainable weights. They are passed an
# initial value which will be assigned when we call:
# {tf.initialize_all_variables().run()}
conv1_weights = tf.Variable(
	tf.truncated_normal([3,3,NUM_CHANNELS,32],
						stddev =STDDEV,
						seed=SEED, dtype=IMAGE_DATA_TYPE))
conv1_biases = tf.Variable(tf.zeros([32],dtype=IMAGE_DATA_TYPE))

conv2_weights = tf.Variable(
	tf.truncated_normal([3,3,32,64],
						stddev =STDDEV,
						seed=SEED, dtype=IMAGE_DATA_TYPE))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64],dtype=IMAGE_DATA_TYPE))

conv3_weights = tf.Variable(
	tf.truncated_normal([3,3,64,64],
						stddev =STDDEV,
						seed=SEED, dtype=IMAGE_DATA_TYPE))
conv3_biases = tf.Variable(tf.constant(0.1, shape=[64],dtype=IMAGE_DATA_TYPE))

conv4_weights = tf.Variable(
	tf.truncated_normal([3,3,64,64],
						stddev =STDDEV,
						seed=SEED, dtype=IMAGE_DATA_TYPE))
conv4_biases = tf.Variable(tf.constant(0.1, shape=[64],dtype=IMAGE_DATA_TYPE))

conv5_weights = tf.Variable(
	tf.truncated_normal([3,3,64,64],
						stddev =STDDEV,
						seed=SEED, dtype=IMAGE_DATA_TYPE))
conv5_biases = tf.Variable(tf.constant(0.1, shape=[64],dtype=IMAGE_DATA_TYPE))

conv6_weights = tf.Variable(
	tf.truncated_normal([3,3,64,128],
						stddev =STDDEV,
						seed=SEED, dtype=IMAGE_DATA_TYPE))
conv6_biases = tf.Variable(tf.constant(0.1, shape=[128],dtype=IMAGE_DATA_TYPE))

conv7_weights = tf.Variable(
	tf.truncated_normal([3,3,128,128],
						stddev =STDDEV,
						seed=SEED, dtype=IMAGE_DATA_TYPE))
conv7_biases = tf.Variable(tf.constant(0.1, shape=[128],dtype=IMAGE_DATA_TYPE))

conv8_weights = tf.Variable(
	tf.truncated_normal([3,3,128,256],
						stddev =STDDEV,
						seed=SEED, dtype=IMAGE_DATA_TYPE))
conv8_biases = tf.Variable(tf.constant(0.1, shape=[256],dtype=IMAGE_DATA_TYPE))

fc1_weights = tf.Variable(
	tf.truncated_normal([IMAGE_SIZE // 16 * IMAGE_SIZE // 16 * 256,1024],
						stddev =STDDEV,
						seed=SEED, dtype=IMAGE_DATA_TYPE))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[1024], dtype=IMAGE_DATA_TYPE))
fc2_weights = tf.Variable(
	tf.truncated_normal([1024, 2*N_POINTS],
						stddev= STDDEV,
						seed=SEED,
						dtype=IMAGE_DATA_TYPE))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[2*N_POINTS], dtype=IMAGE_DATA_TYPE))


def model(data,train=False):
	""" Same padding"""
	# shape matches the data layout: [image index, y, x, depth]
	conv = tf.nn.conv2d(data,
						conv1_weights,
						strides=[1,1,1,1],
						padding='SAME')
	activation = tf.nn.sigmoid(tf.nn.bias_add(conv,conv1_biases))
	pool = tf.nn.max_pool(activation,
						ksize=[1,2,2,1],
						strides=[1,2,2,1],
						padding='SAME')



	conv = tf.nn.conv2d(pool,
						conv2_weights,
						strides=[1,1,1,1],
						padding='SAME')
	activation = tf.nn.sigmoid(tf.nn.bias_add(conv,conv2_biases))
	conv = tf.nn.conv2d(activation,
						conv3_weights,
						strides=[1,1,1,1],
						padding='SAME')
	activation = tf.nn.sigmoid(tf.nn.bias_add(conv,conv3_biases))
	pool = tf.nn.max_pool(activation,
						ksize=[1,2,2,1],
						strides=[1,2,2,1],
						padding='SAME')



	conv = tf.nn.conv2d(pool,
						conv4_weights,
						strides=[1,1,1,1],
						padding='SAME')
	activation = tf.nn.sigmoid(tf.nn.bias_add(conv,conv4_biases))
	conv = tf.nn.conv2d(activation,
						conv5_weights,
						strides=[1,1,1,1],
						padding='SAME')
	activation = tf.nn.sigmoid(tf.nn.bias_add(conv,conv5_biases))
	pool = tf.nn.max_pool(activation,
						ksize=[1,2,2,1],
						strides=[1,2,2,1],
						padding='SAME')




	conv = tf.nn.conv2d(pool,
						conv6_weights,
						strides=[1,1,1,1],
						padding='SAME')
	activation = tf.nn.sigmoid(tf.nn.bias_add(conv,conv6_biases))
	conv = tf.nn.conv2d(activation,
						conv7_weights,
						strides=[1,1,1,1],
						padding='SAME')
	activation = tf.nn.sigmoid(tf.nn.bias_add(conv,conv7_biases))
	pool = tf.nn.max_pool(activation,
						ksize=[1,2,2,1],
						strides=[1,2,2,1],
						padding='SAME')


	conv = tf.nn.conv2d(pool,
						conv8_weights,
						strides=[1,1,1,1],
						padding='SAME')
	activation = tf.nn.sigmoid(tf.nn.bias_add(conv,conv8_biases))
	# Reshape conv layer to all-connected layer
	activation_shape = activation.get_shape().as_list()
	reshape = tf.reshape(activation,
						[activation_shape[0], activation_shape[1]*activation_shape[2]*activation_shape[3]])
	all_connected = tf.nn.sigmoid(tf.matmul(reshape, fc1_weights)+fc1_biases)

	# if train:
	# 	all_connected = tf.nn.dropout(all_connected, 0.8, seed = SEED)
	return tf.matmul(all_connected, fc2_weights) + fc2_biases

# Be careful of the data-structure of network output & labels_node
train_output = model(train_images_node, True)
train_loss =  tf.nn.l2_loss(tf.sub(train_output ,train_labels_node))/(2*N_POINTS*BATCH_SIZE)

# Directly import LeNet L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
				tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
# Add the regularization term to the loss.
regularized_train_loss = train_loss # + LAMDA * regularizers

###############################################################
# Optimizers to be appeared here
# Using simple gradient descent method with constant learning rate

# SGD
# batch = tf.Variable(0, dtype=IMAGE_DATA_TYPE)
# opt = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(regularized_train_loss)

# ADAM
# opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(regularized_train_loss,global_step=batch)

# Mementum

batch = tf.Variable(0, dtype=IMAGE_DATA_TYPE)
# Decay once per epoch, using an exponential schedule starting at marco definition.
learning_rate = tf.train.exponential_decay(
  LEARNING_RATE,                # Base learning rate.
  batch * BATCH_SIZE,  # Current index into the dataset.
  train_size,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)

# Use simple momentum for the optimization.
opt = tf.train.MomentumOptimizer(learning_rate,
                                     0.9).minimize(regularized_train_loss,
                                                   global_step=batch)
################################################################
# Testing code here

# Small utility function to evaluate a dataset by feeding batches of data to
# {test_images_node} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.
test_prediction = model(test_images_node)
tf.add_to_collection('test_prediction', test_prediction)
tf.add_to_collection('test_images_node', test_images_node)
test_loss =  tf.sub(test_prediction,test_labels_node)

#################################################################
# Tensorboard settings

### to be added
##################################################################

def eval_in_batches(data,label, sess):
# Get all test_loss for a dataset by running it in small batches."""
	print(data[:,64,64,:])
	size = data.shape[0]
	if size < EVAL_BATCH_SIZE:
	  raise ValueError("batch size for evals larger than dataset: %d" % size)
	eval_loss = np.ndarray(shape=(size, 2*N_POINTS), dtype=np.float32)
	batch_loss = np.ndarray(shape=(EVAL_BATCH_SIZE, 2*N_POINTS), dtype=np.float32)

	eval_pred = np.ndarray(shape=(size, 2*N_POINTS), dtype=np.float32)
	batch_pred = np.ndarray(shape=(EVAL_BATCH_SIZE, 2*N_POINTS), dtype=np.float32)	

	for begin in xrange(0, size, EVAL_BATCH_SIZE):
	  end = begin + EVAL_BATCH_SIZE
	  if end <= size:
	    eval_loss[begin:end, :], eval_pred[begin:end, :] = sess.run(
	        [test_loss,test_prediction],
	        feed_dict={test_images_node: data[begin:end, ...],
					test_labels_node: label[begin:end]})
	  else:
	    batch_loss[:,:],batch_pred[:,:] = sess.run(
	        [test_loss,test_prediction],
	        feed_dict={test_images_node: data[-EVAL_BATCH_SIZE:, ...],
	        		test_labels_node: label[-EVAL_BATCH_SIZE:]})
	    eval_loss[begin:, :] = batch_loss[begin - size:, :]

	    eval_pred[begin:, :] = batch_pred[begin - size:, :]

	final_eval_loss = (eval_loss**2).mean()/2

	print (eval_pred[:,0])
	return  final_eval_loss


# Model saver

saver = tf.train.Saver()
# Training code here

start_time = time.time()
with tf.Session() as sess:
	# initilalize all parameters in the CNN
	tf.initialize_all_variables().run()
	print('Network initialized')
	# Training loop
	for step in xrange (int(NUM_EPOCHS * train_size) // BATCH_SIZE):
		offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
		batch_data = train_images[offset:(offset + BATCH_SIZE), ...]
		batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
		# This dictionary maps the batch data (as a numpy array) to the
		# node in the graph it should be fed to.
		feed_dict = {train_images_node: batch_data,
					train_labels_node: batch_labels}
					      # Run the graph and fetch some of the nodes.
		_, l, to,lr = sess.run([opt, train_loss, train_output,learning_rate],
						feed_dict=feed_dict)
		# Print one prediction to see if all the predictions are identique
		print (to[:,34])
		if step % EVAL_FREQUENCY == 0:
			elapsed_time = time.time() - start_time
			start_time = time.time()
			print('Step %d (epoch %.2f), %.1f ms' %
					(step, float(step) * BATCH_SIZE / train_size,
					1000 * elapsed_time / EVAL_FREQUENCY))
			print('Minibatch loss: %.5f' % (l))
			eval_fun_out = eval_in_batches(test_images,test_labels,sess)
			print('Validation loss: %.5f' % eval_fun_out)
			sys.stdout.flush()
	print('Validation loss: %.5f' % eval_in_batches(test_images,test_labels,sess))
	saver.save(sess,"./model/sigmoid_250epoch_0001.ckpt")
	print ('Model saved')
