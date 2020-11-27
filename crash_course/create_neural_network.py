from mxnet import nd
from mxnet.gluon import nn

#create first layer of neural network
'''
layer = nn.Dense(2) #fully connected layer with 2 output units
layer.initialize() #initialize layer weight with default initialization method
x = nd.random.uniform(-1,1,(3,4))
w = layer.weight.data() #the weight from initialization
layer(x) #pass a random matrix into the layer to compute the output, it is nd.dot(x, w.transpose()), which is y = xw^T in math form
'''

#chain layers into a neural network
'''
net = nn.Sequential()
##what is Conv2D, MaxPool2D, Dense
net.add( 
    nn.Conv2D(channels=6, kernel_size=5, activation="relu"), #convolutional layer, 6 output channels; kernel, a 5x5 matrix
    nn.MaxPool2D(pool_size=2, strides=2), #maxpool, across 2 dimension
    nn.Conv2D(channels=16, kernel_size=3, activation="relu"),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120, activation="relu"),
    nn.Dense(84, activation="relu"),
    nn.Dense(10)
)
net.initialize()
x = nd.random.uniform(shape=(4,1,28,28))
y = net(x)
'''

#create a neural network flexibly
class MixMLP(nn.Block):
    def __init__(self, **kwargs):
        super(MixMLP, self).__init__(**kwargs)
        self.blk = nn.Sequential()
        # data parse through relu, then undergo np.dot(x, weight^transpose), 
        #
        # result in y1 representing the output of first layer, for nn.Dense(4), it would be relu(y1), then y2 = np.dot(y1, weight2^transpose())
        self.blk.add(
            nn.Dense(3, activation="relu"),
            nn.Dense(4, activation="relu")
        )
        self.dense = nn.Dense(5)
    def forward(self,x):
        y = nd.relu(self.blk(x))
        print(y)
        return self.dense(y)
net = MixMLP()
net.initialize()
x = nd.random.uniform(shape=(2,2))
print(net(x))
