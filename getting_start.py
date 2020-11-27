import matplotlib.pyplot as plt

from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from gluoncv.model_zoo import get_model
'''
url = "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/classification/plane-draw.jpeg"
im_fname = utils.download(url) #downlaod picture from this url
'''

# img = image.imread(im_fname) #read image data, im_fname is mxnet.ndarray.ndarray.NDArray
img = image.imread("horse.jpg") #read image data, im_fname is mxnet.ndarray.ndarray.NDArray
plt.imshow(img.asnumpy()) #convert NDArray to numpy.ndarray
# plt.show()

#transform the picture into model friendly
#resize and crop the image into 32x32
transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(), #Converts an image NDArray or batch of image NDArray to a tensor NDArray
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) #Normalize an tensor of shape (C x H x W) or (N x C x H x W) with mean and standard deviation
])

img = transform_fn(img)
plt.imshow(nd.transpose(img, (1,2,0)).asnumpy())
# plt.show()


#load pre train model
net = get_model('cifar_resnet110_v1', classes=10, pretrained=True)

pred = net(img.expand_dims(axis=0))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
ind = nd.argmax(pred, axis=1).astype('int')
print('The input picture is classified as [%s], with probability %.3f.'%
      (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))