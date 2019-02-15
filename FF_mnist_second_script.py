# -*- coding: utf-8 -*-
#Author: Juan Maroñas Molano 

#Learning pytorch by training using MNIST
#Second script we train feed forward networks and fully connected operator. We start moving from theano based code 
#to Keras based code.

#Things observed in this code:
	#How can we optmize the parameters easily
	#how can we use the torchvision interface to load a provided dataset
	#what is torch.no_grad() and how to use 

#Subject: Neural Networks for Pattern Recognition
#Teacher: Roberto Paredes Palacios
#version 1 Date: November 2017-February 2018
#this version Date: November 2018-February 2019


import torch #main module
if not torch.cuda.is_available():
	print("Buy a gpu")
	exit(-1)
import torchvision #computer vision dataset module
from torchvision import datasets,transforms
from torch import nn #not use in this tutorials. Perfect for weepers and for Keras Users.

import numpy

''' Pytorch Data Loader'''
#This is the basic pipeline of torchvision interface to datasets. Some typycal datasets are directly provided by this interface. However, we can load
#any dataset from folders, just check the documentation or the transfer learning repository in my GitHub. 
#This datasets are loaded using pillow, and thus, any transformation is done in CPU before uploading to GPU. If you are not going to perform data augmentation
#and your dataset and model fit in the gpu, it is probably better to load it directly in GPU, though PyTorch has excellent memory managment. After you have your dataset
#you just feed it into a data loader. Thus, your own dataset must be created as a dataset instance, and then feed it into data.Dataloader. The good point of running things on CPU
#is that you can use several threads to load a batch of data. This is usefull if you perform many transformations  because the transformations are runned in parallel with the main loop. We will cover this in the convolutional tutorial, for the moment just now that this interface is very usefull
mnist_transforms=transforms.Compose([transforms.ToTensor()])
mnist_train=datasets.MNIST('/tmp/',train=True,download=True,transform=mnist_transforms)
mnist_test=datasets.MNIST('/tmp/',train=False,download=False,transform=mnist_transforms)

#this is the dataloader
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=100,shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=100,shuffle=True)


'''Parameters'''
l1=512
l2=512
l3=10
input_d=784
target_d=10
batch=100
epochs=50

w1=torch.from_numpy(numpy.random.normal(0,numpy.sqrt(2.0/float(l1)),(input_d,l1)).astype('float32')).cuda().requires_grad_()
w2=torch.from_numpy(numpy.random.normal(0,numpy.sqrt(2.0/float(l2)),(l1,l2)).astype('float32')).cuda().requires_grad_()
w3=torch.from_numpy(numpy.random.normal(0,numpy.sqrt(2.0/float(l2)),(l2,l3)).astype('float32')).cuda().requires_grad_()

b1=torch.from_numpy(numpy.zeros((1,l1))).cuda().float().requires_grad_()
b2=torch.from_numpy(numpy.zeros((1,l2))).cuda().float().requires_grad_()
b3=torch.from_numpy(numpy.zeros((1,l3))).cuda().float().requires_grad_()#you see here how to cast a torch.Tensor to float32

fRl=nn.ReLU()

'''Optimization'''
optimized_params=[b1,b2,b3,w1,w2,w3]
Loss=nn.CrossEntropyLoss()#softmax plus crossentropy
#new Stuff, We can use an optimizer to perform the updates, instead of doing it in the way I showed in the previous tutorial. 
optimizer=torch.optim.SGD(optimized_params,lr=0.1,momentum=0.9)

for e in range(epochs):
	ce,MC=[0.0]*2
	for x,t in train_loader: #sample one batch
		x,t=x.cuda(),t.cuda()
		x=x.view(-1,784)# This is the drawback of the provided interfaces. Torchvision is the vision package and thus is thinked to be used for Convolutional Networks, so the data is provided as 4D tensor. As I told you you can create your own dataset that returns the data prepared for fully connected models. Again follow the transfer learning tutorial. In this case view is used to reshape the data to have dimension (batch,784)

		predict=torch.mm(fRl(torch.mm(fRl(torch.mm(x,w1)+b1),w2)+b2),w3)+b3 #forward
		o=Loss(predict,t) #compute loss Implicitly apply softmax
		o.backward() #this compute the gradient which respect to leaves. And this is the reason for required gradients True

		#this is new. Remember to call zero_grad to reset the gradients, as pytorch accumulate the gradients in the .grad attribute of Tensor that requires_grad1
		optimizer.step()
		optimizer.zero_grad()
		ce+=o.data

	with torch.no_grad():#it is simple, this deactivates the autograd engine, which is the module that create the computation graph in order to compute the gradients needed for training. Note storing this unsefull gradients save looooooooooooooooot of memory, trust me. With this simple fully connected maybe not, but try and train a deep convolutional model

		#Here I show a way to loop over the test set when using a dataloader (because transformations are required), or when the test set does not fit in memory and has to be loaded from disk-
		for x,t in test_loader:
			x,t=x.cuda(),t.cuda()
			x=x.view(-1,784)
			predict=torch.mm(fRl(torch.mm(fRl(torch.mm(x,w1)+b1),w2)+b2),w3)+b3 #forward

			#note now that I do not apply the softmax, because it does not change the result of the argmax
			index=torch.argmax(predict,1)#compute maximum
			MC+=((index!=t).float().sum()) #accumulate MC error

	print("|| Epoch {} cross entropy {:.5f} and Test error {:.3f}".format(e,ce/600.,100.*MC/10000.))



