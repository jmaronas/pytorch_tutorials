# -*- coding: utf-8 -*-
#Author: Juan Maroñas Molano 

#Learning pytorch by training using CIFAR10
#Sixth script we train convolutional neural network with data autmentation.

#Things observed in this code:
#In this example we will:
	#-Convolutional Neural Network
	#-Data augmentation using the torchvision transform package
	#-learning rate scheduler


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
import os
''' Pytorch Data Loader'''
#This package has several interesting transforms that can be used. This transforms are built on top of pillow. There are other libraries like albumentations that are built on top of opencv and are a bit faster. However I like to use this one because it is made by the same developers as torch. The good points of using pillow is that you can access all the transformations available. The drawback is that these transformations are done in CPU, not in GPU. This is overcomed by the fact that the transformations can be run in parallel in several threads, and that the memory transfer from CPU to GPU is quite efficient due to the way PyTorch manage memory.

#Anyway, you can program your own GPU transformations in case your dataset fits in memory (check my github I made a modification of this package). However, if you need complicated transformations or your data does not fit even in the CPU memory this is the best way to do it. The pipeline is divided into three processes.

#First you should define your transformations. For CIFAR10 we are going to pad, crop and normalize. This can be done in this way. Note that this not depend on the dataset, it is common. In fact you can add your own transformations. I do not cover that in this tutorials because it is very well explained on the internet. Basically you have to manage them in your dataset.

cifar10_transforms_train=transforms.Compose([transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #transforms are different for train and test

cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


#Second, you create your dataset Dataset.  Cifar10 is also provided so we just use it. If you create your own dataset  you can decide how it is loaded to memory and which transformations do you want to apply. Check my tutorial on transfer learning. Basically you use a similar tool to torch.nn but designed for datasets.

workers = (int)(os.popen('nproc').read()) 
cifar10_train=datasets.CIFAR10('/tmp/',train=True,download=True,transform=cifar10_transforms_train)
cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=False,transform=cifar10_transforms_test)

#Third your dataloader. You just pass any dataset you have created. For instance you can decide to shuffle all the dataset at each iteration (that improves generalization) and also yo use several threads. In this case I will detect how many threads does my machine have and use them. Each thread loads a batch of data in parallel to your main loop (your CNN training)
train_loader = torch.utils.data.DataLoader(cifar10_train,batch_size=100,shuffle=True,num_workers=workers)
test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=100,shuffle=False,num_workers=workers)

###############USE THE NN MODULE###################
#lets create a ResNet-18 
class ResNet18(nn.Module):
	def __init__(self):
		super(ResNet18, self).__init__()
		#Pleas note that this code can be done efficient and clear just by using sequentials and for loops. However in this tutorial I want to make things as clear as possible and thus I use an attribute per layer. The pipeline of a Resnet-18 is convolution+bn and then 8 resnet blocks of two convolutions changing the feature map and using stride=2 instead of max pooling. This makes 2*16 + 1 convolution layer-> 17 layer. And the last fully connected layer 17+1=18 layers. Note that this same module can be used for cifar100 only changing the value of n_classes
		self.ReLU=nn.ReLU()
		self.SoftMax=nn.Softmax()
		self.CE = nn.CrossEntropyLoss() #this performs softmax plus cros-entropy		
		self.n_classes=10

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		
		##########
		#RESNET BLOCK 1
		self.b1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.b1_bn1 = nn.BatchNorm2d(64)
		self.b1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.b1_bn2 = nn.BatchNorm2d(64)

		#resnet connection
		#no readaptation is needed

		##########
		#RESNET BLOCK 2
		self.b2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.b2_bn1 = nn.BatchNorm2d(64)
		self.b2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.b2_bn2 = nn.BatchNorm2d(64)
		#resnet connection
		#no readaptation is needed

		##########
		#RESNET BLOCK 3
		self.b3_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
		self.b3_bn1 = nn.BatchNorm2d(128)
		self.b3_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.b3_bn2 = nn.BatchNorm2d(128)

		#resnet connection
		#we need to readapt the input map using 1x1 convolution kernel (like a MLP combining channel dimensions
		self.b3_shortcut_conv= nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
		self.b3_shortcut_bn= nn.BatchNorm2d(128)

		##########
		#RESNET BLOCK 4

		self.b4_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.b4_bn1 = nn.BatchNorm2d(128)
		self.b4_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.b4_bn2 = nn.BatchNorm2d(128)

		#resnet connection
		#no readaptation is needed
		
		##########
		#RESNET BLOCK 5
		self.b5_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
		self.b5_bn1 = nn.BatchNorm2d(256)
		self.b5_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
		self.b5_bn2 = nn.BatchNorm2d(256)

		#resnet connection
		#we need to readapt the input map using 1x1 convolution kernel (like a MLP combining channel dimensions
		self.b5_shortcut_conv= nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
		self.b5_shortcut_bn= nn.BatchNorm2d(256)

		##########
		#RESNET BLOCK 6

		self.b6_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
		self.b6_bn1 = nn.BatchNorm2d(256)
		self.b6_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
		self.b6_bn2 = nn.BatchNorm2d(256)

		#resnet connection
		#no readaptation is needed

		##########
		#RESNET BLOCK 7
		self.b7_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
		self.b7_bn1 = nn.BatchNorm2d(512)
		self.b7_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
		self.b7_bn2 = nn.BatchNorm2d(512)

		#resnet connection
		#we need to readapt the input map using 1x1 convolution kernel (like a MLP combining channel dimensions
		self.b7_shortcut_conv= nn.Conv2d(256,512, kernel_size=1, stride=2, bias=False)
		self.b7_shortcut_bn= nn.BatchNorm2d(512)

		##########
		#RESNET BLOCK 8

		self.b8_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
		self.b8_bn1 = nn.BatchNorm2d(512)
		self.b8_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
		self.b8_bn2 = nn.BatchNorm2d(512)
		#resnet connection
		#no readaptation is needed

		#The only pooling performed
		self.average_pooling=nn.AvgPool2d(4)

		#To connect to the number of classes
		self.Linear=nn.Linear(512,self.n_classes)

	def operator(self,x):
		x=self.ReLU(self.bn1(self.conv1(x)))

		#first resnet block
		x_block=self.b1_bn2(self.b1_conv2(self.ReLU(self.b1_bn1(self.b1_conv1(x)))))
		x=self.ReLU(x+x_block) #resnet connection plus activation

		#second resnet block
		x_block=self.b2_bn2(self.b2_conv2(self.ReLU(self.b2_bn1(self.b2_conv1(x)))))
		x=self.ReLU(x+x_block) #resnet connection plus activation

		#third resnet block
		x_block=self.b3_bn2(self.b3_conv2(self.ReLU(self.b3_bn1(self.b3_conv1(x)))))
		input_readapted=self.b3_shortcut_bn(self.b3_shortcut_conv(x))#we need to readapt the number of maps of the input so it matches the output
		x=self.ReLU(input_readapted+x_block) #resnet connection plus activation

		#fourth resnet block
		x_block=self.b4_bn2(self.b4_conv2(self.ReLU(self.b4_bn1(self.b4_conv1(x)))))
		x=self.ReLU(x+x_block) #resnet connection plus activation

		#fifth resnet block
		x_block=self.b5_bn2(self.b5_conv2(self.ReLU(self.b5_bn1(self.b5_conv1(x)))))
		input_readapted=self.b5_shortcut_bn(self.b5_shortcut_conv(x))#we need to readapt the number of maps of the input so it matches the output
		x=self.ReLU(input_readapted+x_block) #resnet connection plus activation

		#sixth resnet block
		x_block=self.b6_bn2(self.b6_conv2(self.ReLU(self.b6_bn1(self.b6_conv1(x)))))
		x=self.ReLU(x+x_block) #resnet connection plus activation

		#seventh resnet block
		x_block=self.b7_bn2(self.b7_conv2(self.ReLU(self.b7_bn1(self.b7_conv1(x)))))
		input_readapted=self.b7_shortcut_bn(self.b7_shortcut_conv(x))#we need to readapt the number of maps of the input so it matches the output
		x=self.ReLU(input_readapted+x_block) #resnet connection plus activation

		#eigth resnet block
		x_block=self.b8_bn2(self.b8_conv2(self.ReLU(self.b8_bn1(self.b8_conv1(x)))))
		x=self.ReLU(x+x_block) #resnet connection plus activation

		x=self.average_pooling(x)

		x=self.Linear(x.view(x.size(0),-1))

		return x

	def forward_train(self,x):
		self.train()
		return self.operator(x)

	def forward_test(self,x): 
		self.eval()
		return self.operator(x)

	def Loss(self,t_,t):
		return self.CE(t_,t)

#create instance
myNet=ResNet18()
myNet.cuda() #move all the register parameters to  gpus

#lets use a learning rate scheduler. I create my own learning rate scheduler. PyTorch provides several schedulers as well as optimizers. However, I do it in this way to show how it works. We basically create a pointer to a function that returns the actual learning rate.
def lr_scheduler(epoch):
	if epoch < 150:
		return 0.1
	elif epoch < 250:
		return 0.01
	elif epoch < 350:
		return 0.001

scheduler=lr_scheduler
for e in range(350):
	ce_test,MC,ce=[0.0]*3
	#now create and optimizer
	optimizer=torch.optim.SGD(myNet.parameters(),lr=scheduler(e),momentum=0.9)

	for x,t in train_loader: #sample one batch
		x,t=x.cuda(),t.cuda()
		o=myNet.forward_train(x) 
		cost=myNet.Loss(o,t) 
		cost.backward() 
		optimizer.step()
		optimizer.zero_grad()
		ce+=cost.data

	''' You must comment from here'''
	with torch.no_grad():
		for x,t in test_loader:
			x,t=x.cuda(),t.cuda()
			test_pred=myNet.forward_test(x)
			index=torch.argmax(test_pred,1)#compute maximum
			MC+=(index!=t).sum().float() #accumulate MC error
	'''to here, to check how can you run out of memory'''

	'''Uncomment this code after reading appendix and comment the one above'''
	'''
	for x,t in test_loader:
		x,t=x.cuda(),t.cuda()

		o=myNet.forward_test(x) 
		cost=myNet.Loss(o,t)
		ce_test+=cost.data #comment this and uncomment the one without the .data, you will see how the memory explotes
		#ce_test+=cost # you run out of memory
	'''
	print("Epoch {} cross entropy {:.5f} and Test error {:.3f}".format(e,ce/500.,100*MC/10000.))

####################### APPENDIX with explanation #######################
''''
I have already talk about torch.no_grad. However in previous versions this utility did not appear and you might not want to use it for whatever reason. I want to illustrate one key fact of torch variables with this example. Imagine you are measure the calibration error of the test data, measured as the negative log likelihood. You can recover this metric by measuring the cross entropy error, in this particular scenario. So you do a forward through the network and evaluate the cost. This cost is then accumulated in a variable. If you do not use torch.no_grad() or call .data on the tensor you run out of memory. But, why?

It shuld be noted that a torch variable is an instance of class that stores, the data, for instance the value of a loss, and the graph that have created that data, in order to perform automatic differentiation. If we accumulate ce+=o without calling .data we save everything into variable and the memory can explote during the batch optimization. This is not the case, for instance, when doing the forward pass, because when you call backward(), PyTorch automatically dealloc memory for efficience. However, during a test stage we do not perform any backward operation, thus, if we accumulate a differentiable value without accesing .data, our memory can explote. Some values as the classification error are not differentiable and do not suffer from this.

Also, you should consider using torch.no_grad to use a bigger batch during inference. Remark that a bigger batch during inference (which evaluates faster) requires more memory, if you do not use torch.no_grad() you will be creating an unsefull graph. So using torch.no_grad() will allow you to use bigger batch during test stage and thus be faster without saturating your memory.

As an example un comment the above code and check it!

'''

