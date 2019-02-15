# -*- coding: utf-8 -*-
#Author: Juan Maroñas Molano 

#Learning pytorch by training using MNIST
#Fifth script we train feed forward networks and fully connected operator. We use batch normalization and dropout. This is how typical pytorch software looks like.

#Things observed in this code:
#In this example we will:
	#-Introduce Dropout
	#-Use dropout in the weepers (keras users) way.
	#-Use batch normalization
	#-introduce torch.functional

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
class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.ReLU=nn.ReLU()
		self.SoftMax=nn.Softmax()
		self.CE = nn.CrossEntropyLoss() #this performs softmax plus cros-entropy
		self.F1=nn.Linear(784,1024) 
		self.F2=nn.Linear(1024,1024)
		self.FO=nn.Linear(1024,10)
		self.Drop=nn.Dropout(0.5) #easy, use the dropout class (that again inherits from nn.Module)
		self.BN1=nn.BatchNorm1d(1024)
		self.BN2=nn.BatchNorm1d(1024)
		self.BN3=nn.BatchNorm1d(10)
		#A key difference. Because dropouts lacks of parameter we can use only one instance for the whole network. However in batch norm we have the optimizable beta and gamma parameters that must be different for each layer, thus we need a different instance per layer. This is where torch.functional takes part. As I have introduced, calling nn.Dropout, nn.Linear... create instances of classes that inherit from nn.Module, thus they have their __init__, forward... However, we might want to perform only the operation. For instance if a module lacks parameters, like dropout we can apply it by calling nn.functional.dropout() directly in the forward. nn.functional is an interface to the operation a modules performs.

	def forward(self,x):#Maybe you ask your self, how do we control if Drop is creating a mask or multiplying by 0.5? Well it is simple, the self.Drop instance (as it inherits from nn.Module) has its own self.training variable. So just call model.train() or model.eval() to change the behaviour (check the main loop). 
		x=self.ReLU(self.BN1(self.F1(x)))
		x=self.Drop(x)
		x=self.ReLU(self.BN2(self.F2(x)))
		x=nn.functional.dropout(x,0.5,training=self.training)#example of functional
		x=self.BN3(self.FO(x))
		return x

	def forward_train(self,x): #In fact, this is why I like to create another method to infer a label in a test stage, because I usually forget calling .train() or .eval(). You can create the forward_test() and forward_train() in this way. Here dropout applies a random binary mask
		self.train()
		x=self.ReLU(self.BN1(self.F1(x)))
		x=self.Drop(x)
		x=self.ReLU(self.BN2(self.F2(x)))
		x=nn.functional.dropout(x,0.5,training=True)#example of functional
		x=self.BN3(self.FO(x))
		return x

	def forward_eval(self,x): #here dropout multiplies by .5
		self.eval()
		x=self.ReLU(self.BN1(self.F1(x)))
		x=self.Drop(x)
		x=self.ReLU(self.BN2(self.F2(x)))
		x=nn.functional.dropout(x,0.5,training=False)#example of functional
		x=self.BN3(self.FO(x))
		return x

	def Loss(self,t_,t):#we can create the methods we want
		return self.CE(t_,t)

#create instance
myNet=Network()
myNet.cuda() #move all the register parameters to  gpus
for e in range(epochs):
	MC,ce=[0.0]*2
	#now create and optimizer
	optimizer=torch.optim.SGD(myNet.parameters(),lr=0.1,momentum=0.9)
	myNet.train()#change your net to train mode. Basically, calling this changes the flag nn.Module.training to true. So when we are in train mode we apply dropout in the correct way
	for x,t in train_loader: #sample one batch
		x,t=x.cuda(),t.cuda()
		x=x.view(-1,784)
		o=myNet.forward(x) 
		cost=myNet.Loss(o,t) 
		cost.backward() 
		optimizer.step()
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

	print("Epoch {} cross entropy {:.5f} and Test error {:.3f}".format(e,ce/600.,100*MC/10000.))

print("\n\n")
#An equivalent option is:
myNet=Network()
myNet.cuda() #move all the register parameters to  gpus
for e in range(epochs):
	MC,ce=[0.0]*2
	#now create and optimizer
	optimizer=torch.optim.SGD(myNet.parameters(),lr=0.1,momentum=0.9)
	myNet.train()#change your net to train mode. Basically, calling this changes the flag nn.Module.training to true. So when we are in train mode we apply dropout in the correct way
	for x,t in train_loader: #sample one batch
		x,t=x.cuda(),t.cuda()
		x=x.view(-1,784)
		o=myNet.forward_train(x) 
		cost=myNet.Loss(o,t) 
		cost.backward() 
		optimizer.step()
		optimizer.zero_grad()
		ce+=cost.data
 
	myNet.eval()#change to test mode.This set the flag training to False.
	with torch.no_grad():
		for x,t in test_loader:
			x,t=x.cuda(),t.cuda()
			x=x.view(-1,784)
			test_pred=myNet.forward_eval(x)
			index=torch.argmax(test_pred,1)#compute maximum
			MC+=(index!=t).sum().float() #accumulate MC error

	print("Epoch {} cross entropy {:.5f} and Test error {:.3f}".format(e,ce/600.,100*MC/10000.))
