# Pytorch Simple Tutorial

I have made this simple tutorial for the subject of Neural Networks for Pattern Recognition for the Master of Computer Science at Universidad Politécnica de Valencia, so students can quickly go over the  this software and decide which one to use. I go over the BASIC structure and the key important differences between TensorFlow and PyTorch, which make Pytorch, in my opinion, a better choice at least for research. You can check more advanced tutorials that go into more specific details on the different PyTorch functionalities. My tutorials differ from the typical tutorials in the way things are focused. In general, a tutorial is made just to show: "oh how quickly you can get a 99% error rate on mnist". In my tutorial I want to show the structure of PyTorch and how it works, moving from the basic pipeline of how a model is trained in any framework to a PyTorch-based pipeline (which implements this basic pipeline under the hood). This will prevent you from reaching the typicall tutorial local minima in which you use the software in a black-box way. I also provide some important tips specific to this software. 

The tutorial starts from the most basic way to train a neural network. Basically, I train a model in the way we used to do it in Theano, but using Pytorch. Finally, I move to the best way of using the provided PyTorch utilities. The tutorial ends up training a ResNet-18 on CIFAR10.

## Key aspects of PyTorch:

-dynamic graphs

-numpy-like programming

-gpu computation

-class to create your differentiable networks

-easy install and set up and very good support

-very easy to use in comparison with TensorFlow or Theano (in my opinion TensorFlow makes things more complicated). To use TensorFlow I personally prefer Theano (the problem is that Theano is not supported anymore)


# Installation

To run this tutorials you only need to install a virtual enviroment with python3.7 and pytorch 1.0.0 (though I think they can run also in 0.4.0 and python2.7)

```
sudo apt-get install python-virtualenv
sudo apt-get install python3.7 #in ubuntu16 you have to add the snake ppa repository
virtualenv -p python3.7 python3.7_pytorch_cuda10.0_1-0-0
source python3.7_pytorch_cuda10.0_1-0-0/bin/activate
pip install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp37-cp37m-linux_x86_64.whl torchvision
python -c "import torch; print(torch.__version__)"
```
Please use a GPU (google colab if you do not have acces to one).
# Tutorials

The tutorials are organized in different scripts. Just go inside and read the comments. You should follow the next order:

* FF_mnist_first_script.py:
         + How pytorch looks like and the most important methods
         + Typical torch pipeline (without using usefull utilities like torch.nn). Just to show the basic
         + This code resemble what Theano does but with dynamic computation graph
         + The tipical machine learning pipeline: forward, cost, backward, update is done in the most simple way

*  FF_mnist_second_script.py
        + How can we optmize the parameters easily
        + how can we use the torchvision interface to load a provided dataset
        + what is torch.no_grad() and how to use

* FF_mnist_third_script.py
       + This code starts using the nn.Module class provided by pytorch.
       + We will also use the optimizer. However we can always access the parameter list and perform the gradient update as we want. Keras users cannot do this as they do not know how backpropagation works and need an optimizer. Good researchers can invent new update methods and in this case, either they implement their own optimizer or they perform the optimization looping over parameters. Be a researcher and not a keras user.

* FF_mnist_dropout_first_script.py
       + Introduce Dropout
       + Use dropout in the complicated way, however this will let me show the importance of pytorch dynamic graph
  
*  FF_mnist_second_script.py
       + Introduce Dropout
       + Use dropout in the weepers (keras users) way.
       + Use batch normalization
       + Introduce torch.functional

* CONV_cifar10.py
       + Convolutional Neural Network
       + Data augmentation using the torchvision transform package
       + learning rate scheduler






