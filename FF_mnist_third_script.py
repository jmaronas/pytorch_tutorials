# -*- coding: utf-8 -*-
#Author: Juan Maroñas Molano 

#Learning pytorch by training using MNIST
#Third script we train feed forward networks and fully connected operator. We start using some good stuff provided by pytorch.

#Things observed in this code:
	#-This code starts using the nn.Module class provided by pytorch.
	#- We will also use the optimizer. However we can always access the parameter list and perform the gradient update as we want. Keras users cannot do this as they do not know how backpropagation works and need an optimizer. Good researchers can invent new update methods and in this case, either they implement their own optimizer or they perform the optimization looping over parameters. Be a researcher and not a keras user.

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
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=100,shuffle=False)

batch=100
input_d=784

###############USE THE NN MODULE###################
#The nn.Module is a class provided by  pytorch that have fantastic properties. Just refer to the documentation, here I only expose some of them.
#We have to override two methods, the __init__ and the forward. The __init__ is used to define the parameters and the forward to perform the
#forward operation of the network. 

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.ReLU=nn.ReLU()
		self.SoftMax=nn.Softmax(dim=1)
		self.CE = nn.CrossEntropyLoss() #this performs softmax plus cross-entropy loss
		self.F1=nn.Linear(784,512) #This will create a Linear operator, i.e it creates a matrix weight and a vector bias. These new torch.Tensors with required_grad=True are automatically incorporated as part of the parameter list. The parameter self.F1 is overrided if you now do: self.F1=nn.Linear(512,512). Creating dynamic networks (for instance a class Network that implement a fully connected with a variable depth or different topology) can be done using utilities from the nn.Module such as register list or register module, check them.  One key point, nn.Linear is a nn.Module also, with its __init__ and forward overrided. This means you can create your own operator and then incorporate it in other modules (this is what people do to create Residual Networks, for instance).

		#On the other hand, we can create our own operations by defining the parameters. For instance we want to create a second layer of 512 hidden units. We can do it this way:
		#First create the parameters needed as nn.Parameters
		self.w=nn.Parameter(torch.from_numpy(numpy.random.randn(512,512)/numpy.sqrt(512)).float())
		self.b=nn.Parameter(torch.zeros(512,).float())
		#Parameter is  a torch.Tensor that requires_grad. The ONLY difference is that the nn.Module automatically add it to the parameter list. Again the name of the variables
		#must be different if you want to register several of them.
		#The __init__ method from nn.Linear defines these variables, but with uniform samplinginitialization.

		#All we have seen can be mix. Either we use a built in layer like nn.Linear (which creates a weight and a bias). Or we can create ours, like with self.w and self.b. Now we use a built in, because we want. The problem is that we might want to change the default initialization. Well, just access the parameters.
		self.FO=nn.Linear(512,10)
		#If you want to access the parameters of the Linear Module you can easily do it in this way.
		#Change initialization from F1 and FO. It use uniform distribution by default, however we want to use Gaussian.
		self.F1.weight.data=torch.from_numpy(numpy.random.randn(512,784)/numpy.sqrt(512)).float() 
		self.FO.weight.data=torch.from_numpy(numpy.random.randn(10,512)/numpy.sqrt(10)).float()
		#Flip dimension. This has to do with blas and cublas library that pytorch uses as backend.

	def forward(self,x):	
		#we can override x no problem. Pytorch can see they are different variables
		x=self.ReLU(self.F1(x))
		x=self.ReLU(torch.mm(x,self.w)+self.b) #this is what nn.Linear do inside. The forward() method from nn.Linear does this operations
		x=self.FO(x)
		return x

	def inference(self,x):#I usually create this method to return sofmax directly or to control other stuff I will show in other tutorial
		x=self.ReLU(self.F1(x))
		x=self.ReLU(torch.mm(x,self.w)+self.b)
		x=self.SoftMax(self.FO(x))
		return x

	def Loss(self,t,t_pred):#we can create the methods we want
	        return self.CE(t_pred,t)


#create instance
myNet=Network()
myNet.cuda() #move all the registered nn.Parameters and torch.tensor to the gpus,i.e, it moves to gpu everything that involves computation.
for e in range(epochs):
	MC,ce=[0.0]*2
	#now create and optimizer. Calling myNet.parameters() returns a list with all the registered parameters. This is the list of parameters that I referred to when I was creating the nn.Module. Each nn.Parameter is added to this list, and you can access it directly in order to optimize wrt the parameters.
	optimizer=torch.optim.SGD(myNet.parameters(),lr=0.1,momentum=0.9)
	for x,t in train_loader: #sample one batch
		x,t=x.cuda(),t.cuda()
		x=x.view(-1,784)
		o=myNet.forward(x) #forward. o has to be the pre-softmax because the cross entropy loss applies it.
		o=myNet.Loss(t,o) #compute loss
		o.backward() #this compute the gradient which respect to leaves. And this is the reason for required gradients True
		optimizer.step()#step in gradient direction
		optimizer.zero_grad()
		ce+=o.data
 
	with torch.no_grad():
		for x,t in test_loader:
			x,t=x.cuda(),t.cuda()
			x=x.view(-1,784)
			test_pred=myNet.inference(x)
			index=torch.argmax(test_pred,1)#compute maximum
			MC+=((index!=t).sum().float()) #accumulate MC error

	print("Cross entropy {:.3f} and Test error {:.3f}".format(ce/600.,100*MC/10000.)) 
	
