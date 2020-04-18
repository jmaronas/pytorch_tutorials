# -*- coding: utf-8 -*-
# Author: Juan Maroñas Molano (PRHLT UPV) (jmaronasm@gmail.com)

# How does PyTorch compute derivatives


import torch

###################
#### Example 1 ####
## Compute the gradients of a scalar function
## L = y1+y2 -> we compute dL/dx1 and dL/dx2 where x1 and x2 are the inputs in just one call

x = torch.tensor([1.,2.],requires_grad = True)
W = torch.tensor([[1., 2.],[3., 4.]])

y = x@W
L = y.sum()

print(x)
print(W)
print(y)
print(L)
L.backward() # Return the derivative of L = y1+y2 w.r.t the inputs by applying chain rule.
             # x.grad will have dL/dx1=dL/dy1*dy1/dx1 + dL/dy2*dy2/dx1 and dL/dx2 

print(x.grad) # contains the gradients dL/dx1= dL/dy1*dy1/dx1 + dL/dy2*dy2/dx1 in x.grad[0] and dL/dx2 in x.grad[1]

## Equivalent alternatives using torch.autograd:
# torch.autograd.backward(y.sum()) #
# torch.autograd.backward(y.sum(),torch.tensor(1.)) # we will see soon what torch.tensor(1.0) means
# x_grad=torch.autograd.grad(y.sum(),x)
# x_grad=torch.autograd.grad(y.sum(),x,torch.tensor(1.))

## The difference between grad and backward() is that backward() computes the gradients w.r.t the leaves of the graph while grad only compute it w.r.t the inputs

print("---------------------")
print("---------------------")
print("---------------------")

###################
#### Example 2 ####
## Compute the gradients of a function by computing the gradients of each of the gradients of the final sum separately.
## L = y1+y2 -> we compute dL/dx1  and dL/dx2 by manually computing dy1/dx1 dy2/dx1 and then summing them. Same for x2

x = torch.tensor([1.,2.],requires_grad = True)
W = torch.tensor([[1., 2.],[3., 4.]])

y = x@W

print(x)
print(W)
print(y)

binary = torch.tensor([1.,0])

x_grad = torch.autograd.grad(y,x,binary,retain_graph = True) # this torch binary vector tells pytorch which output from the vector y must be differentiated.
                                                             # So in this example the first position from binary tells pytorch to compute (1) or not compute (0)
                                                             # the derivative of y1 w.r.t the inputs.
                                                             # This call outputs a vector x_grad where the first position is dy1/dx1 and the second is dy1/dx2.
                                                             # We use retain_graph = True because if not pytorch deallocate memory during the backward pass so the graph
                                                             # is destroyed
binary = torch.tensor([0.,1.])
x_grad = x_grad[0] + torch.autograd.grad(y,x,binary,retain_graph = True)[0] # with this you compute the total gradient
print(x_grad)

## A clear alternative to this is to do
binary = torch.tensor([1.,1.])
x_grad = torch.autograd.grad(y,x,binary)[0] 

# More alternatives include (extrapolating to any of the above calls to grad)
# y.backward(binary)

print("---------------------")
print("---------------------")
print("---------------------")


################### 
#### Example 3 ####
## Compute the Jacobian Matrix
## The Jacobian remember is a matrix of partial derivatives. In this case dy1/dx1 dy2/dx1
##                                                                        dy1/dx2 dy2/dx2
## So we have seen that we can compute the partial derivatives with a binary vector. This binary vector indicates the output dimension being differentiated w.r.t inputs.
## So passing [0.,1.] means that we compute dy2/dx1 and dy2/dx2 which will be saved in x.grad 
## Thus computing the Jacobian can be done in two calls to backward.


Jacobian = torch.zeros((2,2),dtype=float)
x = torch.tensor([1.,2.],requires_grad = True)
W = torch.tensor([[1., 2.],[3., 4.]])
y = x@W

y.backward(torch.tensor([1.,0.]),retain_graph = True) # Derivatives of y1 w.r.t x1 and x2
Jacobian[:,0] = x.grad
x.grad.zero_() # reset the gradient because by default pytorch always accumulates the gradient

y.backward(torch.tensor([0.,1.])) # we can free memory now
Jacobian[:,1] = x.grad

print(Jacobian)

# Alternatives
# torch.autograd.backward(y,torch.tensor([0.,1.]))
# Jacobian[:,1] = x.grad 
# x_grad = torch.autograd.grad(y,x,torch.tensor([1,0.])) # equivalent
# Jacobian[:,0] = x_grad 

## Note that the Jacobian gives the total derivative if we sum the colums

print(Jacobian.sum(1))





