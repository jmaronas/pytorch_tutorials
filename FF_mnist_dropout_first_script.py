# -*- coding: utf-8 -*-
#Author: Juan Maroñas Molano 

#Learning pytorch by training using MNIST
#Fourth script we train feed forward networks and fully connected operator. Introduce dropout.

#Things observed in this code:
#In this example we will:
	#-Introduce Dropout
	#-Use dropout in the complicated way, however this will let me show the importance of pytorch dynamic graph.

#Subject: Neural Networks for Pattern Recognition
#Teacher: Roberto Paredes Palacios
#version 1 Date: November 2017-February 2018
#this version Date: November 2018-February 2019


import torch #main module
if not torch.cuda.is_available():
	print("unable to run on GPU")
	exit(-1)

import torchvision #computer vision dataset module
from torchvision import datasets,transforms
from torch import nn #Keras users will now be really happy

import numpy
'''
Specific parameters of network
'''
epochs=10
batch=100
data_len=60000

''' Pytorch Data Loader'''
mnist_transforms=transforms.Compose([transforms.ToTensor()])
mnist_train=datasets.MNIST('/tmp/',train=True,download=True,transform=mnist_transforms)
mnist_test=datasets.MNIST('/tmp/',train=False,download=False,transform=mnist_transforms)

#this is the dataloader
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=100,shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=100,shuffle=True)

batch=100
input_d=784


###############USE THE NN MODULE###################
#The module is the same as in the previous example, however we have to incorporate a dropout operator. In this case I will use Bernouilli dropout. We use dropout with probability 0.5
class Network(nn.Module): 
	def __init__(self):
		super(Network, self).__init__()
		self.ReLU=nn.ReLU()#see functional, usefull if we do not need to register parameter
		self.SoftMax=nn.Softmax()
		self.CE = nn.CrossEntropyLoss() #this performs softmax plus cros-entropy
		self.F1=nn.Linear(784,512)
		self.FO=nn.Linear(512,10)

		#dropout. We need a bernouilli distribution to generate a binary mask with probability given by drop probability and then cut out the post activation
		self.drop_factor=0.5*torch.ones((512,)).cuda() #Important, required_grad can be False because the dropout operation has no parameter. We create two tensors here to preallocate memory, instaed of doing it during the forward. With, this, in the forward pass we only need to sample the mask.  We store this in gpu as .cuda() only moves parameters
		#preallocate memory
		self.drop_var=torch.zeros((512,)).cuda()

	def forward(self,x):	
		#sample the drop mask if we are in a training phase
		if self.training: #you will understand the self.training when running the main loop
			self.drop_var.data.bernoulli_(self.drop_factor)
		x=self.ReLU(self.F1(x))
		if self.training:#self.training is a boolean variable
			x=x*self.drop_var #in train we apply dropout dropout
		else:
			x=x*self.drop_factor #in test we multiply by dropout probability
		x=self.FO(x)
		return x

	def Loss(self,t_,t):
		return self.CE(t_,t)


#create instance

myNet=Network()
myNet.cuda() #move everything to gpu.

#you can see how the drop variables does not appear in the parameter list. This is because they are not wrapped with nn.Parameter. It makes sense to do it in this way because dropout operator does not have any parameter
for name,param in myNet.named_parameters():
	print("Parameter name {} parameters in cuda {}".format(name,param.is_cuda))

#you can check how calling to .cuda() moves everything to cuda.
print("Dropout parameters are in cuda??. drop factor is in cuda {}   drop_var is in cuda {}".format(myNet.drop_factor.is_cuda,myNet.drop_var.is_cuda))

for e in range(epochs):
	MC,ce=[0.0]*2
	#now create and optimizer
	optimizer=torch.optim.SGD(myNet.parameters(),lr=0.1,momentum=0.9)
	myNet.train()#change your net to train mode. Basically, calling this changes the flag nn.Module.training to true. So when we are in train mode we apply dropout in the correct way
	for x,t in train_loader: #sample one batch
		x,t=x.cuda(),t.cuda()
		x=x.view(-1,784)
		o=myNet.forward(x) #forward
		cost=myNet.Loss(o,t) 
		cost.backward() #this compute the gradient which respect to leaves. And this is the reason for required gradients True
		optimizer.step()#step in gradient direction
		optimizer.zero_grad()
		ce+=cost.data
 
	myNet.eval()#change to test mode.This set the flag training to False.
	with torch.no_grad():
		for x,t in test_loader:
			x,t=x.cuda(),t.cuda()
			x=x.view(-1,784)
			test_pred=myNet.forward(x)
			index=torch.argmax(test_pred,1)#compute maximum
			MC+=(index!=t).sum().float() #accumulate MC error

	print("Cross entropy {:.3f} and Test error {:.3f}".format(ce/600.,100*MC/10000.))


################## APPENDIX with explanation ##################
'''

In this example we can start understanding how dynamic graphs make things easier. In theano or in tensorflow you have to create two different graphs, one that evaluates dropout in test mode and one in training mode. I have to say that this can be done efficient, there is no need to create two different graphs, you can create only one graph and then make to different paths when dropout appears. I think that in Theano this was done with some if-like statement. I do not remember, I leaved theano more than one year ago. Anyway, in pytorch it can be done with the traditional if statement, and that facilitates our live. Moreover, you can still see some example through the internet where people create two graphs in TensorFlow, one for training and one for test. In pytorch is simpler. Because the graph is created online and does not need to be compile first you can change the pipeline of operations in the way you want at any moment. Imagine you want to do something different starting from iteration 100. In theano you will have to create a new graph that shares the variables with the initial one, and make the desired modifications. Here you can just use an if statement.

'''
		


