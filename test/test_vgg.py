"""
This is the file to test whether the VGG model converted from Torch Lua is identical to the original model.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import vgg_normalised  # vgg_normalised is the PyTorch model converted from Torch Lua.

CIFAR_NUMPY = '/data/mixup/cifar10/data_batch_1_data.npy'  # The 1/5 numpy file for CIFAR.

parser = argparse.ArgumentParser(description='Test VGG')
parser.add_argument('--image_id', default=0, type=int, help='image id')
args = parser.parse_args()


def test_vgg(this_vgg, image, this_torch_feature):
    """Test the VGG model and compare with the Torch Lua model output.

    :param this_vgg: A PyTorch model.
    :param image: A PyTorch Tensor as the image.
    :param this_torch_feature: A numpy as the output of the torch lua model.
    :return: None
    """
    this_vgg.cuda()
    image = image.cuda()
    feature = this_vgg(image)
    feature = feature.data.cpu().numpy()  # feature has shape 1x512x16x16
    # Compare feature with torch feature.
    for i in range(feature.shape[1]):
        for j in range(feature.shape[2]):
            for k in range(feature.shape[3]):
                if feature[0][i][j][k] != this_torch_feature[i][j][k]:
                    print('{},{},{}:{}/{}'.format(i, j, k, feature[0][i][j][k], this_torch_feature[i][j][k]))


def load_vgg_model():
    """Load the VGG model."""
    this_vgg = vgg_normalised.vgg_normalised
    this_vgg.load_state_dict(torch.load('vgg_normalised.pth'))
    """
    This is to ensure that the vgg is the same as the model used in PyTorch lua as below:
    vgg = torch.load(opt.vgg)
    for i=53,32,-1 do
        vgg:remove(i)
    end
    This actually removes 22 layers from the VGG model.
    """
    this_vgg = nn.Sequential(*list(this_vgg)[:-22])
    this_vgg.eval()
    return this_vgg


def load_image_from_cifar(image_id):
    """This is to load the image from CIFAR numpy and pre-process.

    :param image_id: An integer as the image id to load from CIFAR.
    :return img: A PyTorch Tensor.
    """
    cifar = np.load(CIFAR_NUMPY)
    image = cifar[image_id]

    image = image / 255.0  # Normalize
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)  # Change from 3D to 4D
    image = image.float()
    image = Variable(image, volatile=True)
    return image


if __name__ == '__main__':
    vgg = load_vgg_model()
    img = load_image_from_cifar(args.image_id)
    torch_feature = np.load('torch_feature.npy')  # Load Torch Feature.
    test_vgg(vgg, img, torch_feature)
