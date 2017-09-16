import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
# import scipy.misc
from util import *

def MnistNetwork(input,keep_prob,scope='Mnist',reuse = False):
	with tf.variable_scope(scope,reuse = reuse) as sc :
		with slim.arg_scope([slim.conv2d,slim.fully_connected],
										  biases_initializer=tf.constant_initializer(0.0),
										  activation_fn=tf.nn.relu):
			net = slim.conv2d(input, 32, [5, 5], scope='conv1')
			net = slim.max_pool2d(net,[2, 2], 2, scope='pool1')
			net = slim.conv2d(net, 64, [5, 5], scope='conv2')
			net = slim.max_pool2d(net,[2, 2], 2, scope='pool2')
			net = slim.flatten(net)
			net = slim.fully_connected(net,1024,scope='fc1')
			net = tf.nn.dropout(net, keep_prob)
			net = slim.fully_connected(net,10,activation_fn=None,scope='fc2')
			net = tf.nn.softmax(net)
			return net


def loss(prediction,output):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(output * tf.log(prediction), reduction_indices=[1]))
	correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(output,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return cross_entropy,accuracy

def getAdversarial(x,grad,eps) :
	t1 = eps*tf.cast(grad > 0,dtype = tf.float32)
	t2 = -eps*tf.cast(grad < 0,dtype = tf.float32)
	return x + t1+t2

eps = 0.1
alpha = 0.5
with tf.Graph().as_default():
		
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	keep_prob = tf.placeholder(tf.float32)
	x_image = tf.reshape(x, [-1,28,28,1])
	y_conv=MnistNetwork(x_image,keep_prob)

	cross_entropy,accuracy=loss(y_conv,y_)
	grads = tf.gradients(cross_entropy,x)

	x_adv = tf.reshape(getAdversarial(x,grads[0],eps),[-1,28,28,1])

	y_conv_mod=MnistNetwork(x_adv,keep_prob,reuse = True)

	cross_entropy_mod,_=loss(y_conv_mod,y_)

	model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mnist')

	grad1 = tf.gradients(cross_entropy,model_vars)
	grad2 = tf.gradients(cross_entropy_mod,model_vars)
	total_grad = []
	for i in range(len(grad1)) :
		total_grad.append(alpha*grad1[i] + (1-alpha)*grad2[i])
	
	trainer = tf.train.AdamOptimizer(1e-4)
	train_step = trainer.apply_gradients(zip(total_grad,model_vars))
	

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	for i in range(7500):
	  batch = mnist.train.next_batch(50)
	  if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	test_data =  mnist.test.images
	test_labels = mnist.test.labels


	print("test accuracy %g"%accuracy.eval(feed_dict={
		x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
	
	
	modified_images = np.genfromtxt('modifiedImData.txt')
	test_labels = np.genfromtxt('modifiedImLabel.txt')
	print("test accuracy over adversarial examples is %g"%accuracy.eval(feed_dict={
		x: modified_images, y_: test_labels, keep_prob: 1.0}))	


	
