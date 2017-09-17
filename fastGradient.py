import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
# import scipy.misc
from util import *

def MnistNetwork(input,keep_prob,scope='Mnist'):
	with tf.variable_scope(scope,[input]) as sc :
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

def generateAdversarialExamples(grad_op,test_data,test_labels,eps) :
	test_grads = sess.run(grad_op,feed_dict = {x:test_data,y_:test_labels,keep_prob: 1.0})[0]
	modified_images = []
	for i in range(len(test_grads)) :
		im = test_data[i]
		grad_im = test_grads[i]
		t1 = eps*(grad_im>0)
		t2 = -eps*(grad_im<0)
		im_mod = im +t1 + t2
		modified_images.append(im_mod)
	modified_images = np.array(modified_images)
	return modified_images

def getMisClassificationConfidence(logits,labels) :
	logit_max = tf.cast(tf.reduce_max(logits,1),tf.float32)
	unequals = tf.cast(tf.not_equal(tf.argmax(logits,1),tf.argmax(labels,1)),tf.float32)
	confidence = logit_max * unequals
	sum_confidence = tf.reduce_sum(confidence)
	total = tf.reduce_sum(unequals)
	return tf.cond(tf.equal(total,tf.constant(0.0)),lambda: tf.constant(0.0),lambda: sum_confidence/total)

with tf.Graph().as_default():
		
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	keep_prob = tf.placeholder(tf.float32)
	x_image = tf.reshape(x, [-1,28,28,1])
	y_conv=MnistNetwork(x_image,keep_prob,scope='Mnist')

	cross_entropy,accuracy=loss(y_conv,y_)
	miss_class_conf = getMisClassificationConfidence(y_conv,y_)
	grads = tf.gradients(cross_entropy,x)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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


	test_acc,mis_class = sess.run([accuracy,miss_class_conf],feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
	print("test accuracy is %g and miss classification confidence is %g"%(test_acc,mis_class))

	
	eps = 0.1
	
	modified_images = generateAdversarialExamples(grads,test_data,test_labels,eps)
	np.savetxt('modifiedImData.txt',modified_images)
	np.savetxt('modifiedImLabel.txt',test_labels)

	test_acc,mis_class = sess.run([accuracy,miss_class_conf],feed_dict={x: modified_images, y_: test_labels, keep_prob: 1.0})
	print("test accuracy over adversarial examples is %g and mis classification confidence is %g"%(test_acc,mis_class))


	original_samples = np.reshape(test_data[0:196],[-1,28,28,1])
	modified_samples = np.reshape(modified_images[0:196],[-1,28,28,1])
	save_images(original_samples, image_manifold_size(original_samples.shape[0]), 'originalImages.png')
	save_images(modified_samples, image_manifold_size(modified_samples.shape[0]), 'adversarialImages.png')
