"""A PyTorch version of AdaIN Style Transfer."""
import argparse

import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn as nn

import AdaptiveInstanceNormalization as Adain
from models import decoder, vgg_normalised


parser = argparse.ArgumentParser(description='Run AdaIN style transfer on CIFAR.')
parser.add_argument('--alpha', default=0.75, type=float, help='The weight of style feature.')
args = parser.parse_args()

CIFAR_NUMPY = '/data/mixup/cifar10/data_batch_1_data.npy'  # The 1/5 numpy file for CIFAR.


def style_transfer():
    """Style transfer between content image and style image."""
    style_feature = vgg(style_img)  # torch.Size([1, 512, 16, 16])
    content_feature = vgg(content_img)  # torch.Size([1, 512, 16, 16])
    input = torch.cat((content_feature, style_feature), 0)
    adain = Adain.AdaptiveInstanceNormalization()
    target_feature = adain(input)
    target_feature = args.alpha * target_feature + (1 - args.alpha) * content_feature
    return decoder(target_feature)


def load_vgg_model():
    """Load the VGG model."""
    this_vgg = vgg_normalised.vgg_normalised
    this_vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
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
    image = CIFAR[image_id]

    image = image / 255.0  # Normalize
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)  # Change from 3D to 4D
    image = image.float()
    image = Variable(image, volatile=True)
    return image


def load_decoder_model():
    """Load the decoder model which is converted from the Torch lua model using
    git@github.com:clcarwin/convert_torch_to_pytorch.git.

    :return: The decoder model as described in the paper.
    """
    this_decoder = decoder.decoder
    this_decoder.load_state_dict(torch.load('models/decoder.pth'))
    this_decoder.eval()
    return this_decoder


if __name__ == '__main__':
    vgg = load_vgg_model()
    decoder = load_decoder_model()
    CIFAR = np.load(CIFAR_NUMPY)
    content_img = load_image_from_cifar(4)
    for i in range(10):
        style_img = load_image_from_cifar(i)
        style_transfer_res = style_transfer()
        style_transfer_res = style_transfer_res.data.numpy()
        style_transfer_res[style_transfer_res < 0] = 0
        style_transfer_res[style_transfer_res > 1] = 1
        scipy.misc.imsave('res/img_4_stylized_img_{}.jpg'.format(i), np.moveaxis(style_transfer_res.squeeze(0), 0, -1))
