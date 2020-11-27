from mxnet import nd
from mxnet import autograd

#we want to differentiate a function f(x) = 2x^2 with respect to parameter x
x = nd.array([[1,2],[3,4]]) #initialize value of x
x.attach_grad() #tell ndarray we plan to store a gradient in MXNET

#we need to store y into MXNET
with autograd.record():
    y = 2*x*x

y.backward() #invoke back propagation
x.grad #get gradient of x