# -*- coding: utf-8 -*-
#Author: Juan Maroñas Molano 

#Learning pytorch by training using MNIST
#Start with feed forward networks and fully connected operator

#Things observed in this code:
	#How pytorch looks like and the most important methods
	#Typical torch pipeline (without using usefull utilities like torch.nn). Just to show the basic
	#This code resemble what Theano does but with dynamic computation graph
	#The tipical machine learning pipeline: forward, cost, backward, update is done in the most simple way

#Subject: Neural Networks for Pattern Recognition
#Teacher: Roberto Paredes Palacios
#version 1 Date: November 2017-February 2018
#this version Date: November 2018-February 2019


import torch #main module
if not torch.cuda.is_available():
	print("Buy a gpu")
	exit(-1)
import torch.utils.data
import torch.nn as nn
import numpy #numpy module

'''
Specific parameters of network
'''
epochs=10
batch=100
data_len=60000

'''
LOAD MNIST DATASET
'''
#download dataset from Montreal webpage. http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz. Place it in /tmp/
dataset='/tmp/mnist.pkl.gz'
import gzip
import pickle
try:
	with gzip.open(dataset, 'rb') as f:
		try:
			train_set_a, valid_set_a, test_set = pickle.load(f, encoding='latin1')
		except:
			in_set_a, valid_set_a, test_set = pickle.load(f)
except:
	raise NameError("Execute in the terminal the next command: wget http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz -P /tmp/")


train_feat=numpy.zeros((60000,784),dtype='float32')
train_labl=numpy.zeros((60000,),dtype='int64')

train_feat[0:50000,:]=train_set_a[0].astype('float32')
train_feat[50000:,:]=valid_set_a[0].astype('float32')
train_labl[0:50000]=train_set_a[1].astype('int64')
train_labl[50000:]=valid_set_a[1].astype('int64')#labels in pytorch must be int64


test_feat=torch.from_numpy(test_set[0].astype('float32')).cuda() #Create two variables. We can use torch.from_numpy to wrap a numpy array to torch, then use .cuda to move to GPU. The good point of new versions of pytorch is that a torch tensor directly support automatic differentiation (previously we need to wrap it with variable)
test_labl=torch.from_numpy(test_set[1].astype('int64')).cuda()

'''
Load into pytorch loader
'''
#pytorch loader can do many fancy things. We will see in the convolution tutorial. For the moment we just create a TensorDataset with our data. 
mnist_dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_feat).cuda(), torch.from_numpy(train_labl).cuda())
#Now create a Loader with this data set. We could iterate over this and randonmly sample (shuffle=True) a batch.
train_loader = torch.utils.data.DataLoader(mnist_dataset_train,batch_size=100,shuffle=True)

'''
Define functions
'''
#Define a function with the typical activations. This activations can be obtained from the nn.Module. 
def activation(x,tip):
	if tip=="sof":
		f=nn.Softmax(dim=1)
		return f(x)
	elif tip=='relu':
		f=nn.ReLU()
		return f(x)
	else:
		print("activation function not present")
		exit(-1)

#forward operation. 
def forward(x):
	x2 = activation(torch.mm(x,w1)+b1,'relu')
	y_pre_activation = torch.mm(x2,w2)+b2
	return y_pre_activation 

#inference. We can use the forward function, however we explicitly apply softmax here though it is not necessary to compute the argmax.
#In the next tutorials I cover when, in my opinion, I like make this difference explictly. A reason could be if you want to directly return the softmax output.
#This is because, in pytorch, the crossentropy loss applies this activation
def inference(x):
	x2 = activation(torch.mm(x,w1)+b1,'relu')
	preactivation =activation( torch.mm(x2,w2)+b2,'sof')
	return preactivation

#loss
CE = nn.CrossEntropyLoss() #this performs softmax plus cros-entropy
def loss(t_,t):
	return CE(t_,t)

#define your layer dimension. We train a fully connected neural network of only one layer with 512 neurons
l1=512
l2=10
input_d=784
target_d=10

#Create parameter of the network (shared variables in theano). Requires gradient is important (see apendix A). By default this attribute is set to false, unless is a nn.Parameter (we will cover this in other tutorials)
w1=torch.from_numpy(numpy.random.normal(0,numpy.sqrt(2.0/float(l1)),(input_d,l1)).astype('float32')).cuda().requires_grad_()
w2=torch.from_numpy(numpy.random.normal(0,numpy.sqrt(2.0/float(l2)),(l1,l2)).astype('float32')).cuda().requires_grad_()
b1=torch.zeros((1,l1)).cuda().requires_grad_()
b2=torch.zeros((1,l2)).cuda().requires_grad_()
differentiated_params=[w1,w2,b1,b2] #our list with parameters to update
epochs=20

#and now the main loop: forward, cost, backward, update, reset grads
for e in range(epochs):
	ce=0.0
	for x,t in train_loader: #sample one batch
		x,t=x,t.cuda()#move to gpu (when variables are one dimensional, such as t, the torch.dataloader move them to CPU, I do not know why). 
		predict=forward(x) #forward
		o=loss(predict,t) #compute loss
		o.backward() #this compute the gradient which respect to leaves. And this is the reason for required gradients True.  It sould be note that o store the data but also the graph of the variables that have computed this value, in order to perform automatic differentiation. This is the key difference with TensorFlow, we do not need to compile a graph before predicting a value, we just do it online and the graph is created automatically when we perform the different operations.
		ce+=o.data#store the ce for printing.

		for p in differentiated_params: #loop over params (this will be the update we pass to Theano function)
			p.data=p.data-0.01*p.grad.data
			p.grad.data.zero_() # and then reset the gradient
			#Adding momentum should be straightforward. Update gradient and old momentum into weight. Update old momentum with the last update, then zero grad the gradients

	test_pred=inference(test_feat)#compute the softmax from test data, though it is not necessary unless you want to have access to the class posterior probability assigned by the model
	index=torch.argmax(test_pred,1).data#compute argument maximum
	MC=(index!=test_labl).float().mean()*100 #compute MC error in %
	print(("Epoch {} cross entropy {:.5f} and Test error {:.3f}").format(e,ce/600.,MC))

################## APPENDIX with explanation ##################
'''
-PyTorch have two ways of computing gradients. You can call the method grad(x,y) and it wil compute the gradient of x wrt y. 

-Doing this can be tedious when having lots of parameters, as we need a backtrace of what has been done. The backward method compute the gradient of the caller which respect to all the leaves in the graph that requires grad. For example for adding adversarial noise the inputs should require grad. This is the easiest way, just compute the output given the input and call backward method.

-Also. The key difference with Theano or Tensorflow is that the graph is created dynamically and not statically. This have powerfull advantages, as we will discuss in other tutorials. For instance we can have access to the output of any layer in a neural network directly, without having to create a graph that outputs this value. The good point of static graph is that once created it can be optimized. The good point of dynamic graph is that you can program in the same way you are used to.

-Memory transfers in pytorch are not so harmfull as pytorch use a very efficient memory managment based on memory pools, to avoid malloc and realloc continously. However if we can avoid transfering data through the PCI that will be fine. This is why I store the dataset in the GPU once created, because it fits. We will cover this in more detail when talking about big datasets or transformations

-If we access the .data attribute we recover only the data. I will explain this in the next tutorial.

-https://github.com/jcjohnson/pytorch-examples You can find here why this kind of softwares are so powerfull. 
'''
