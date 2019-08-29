# python3.7
"""Retrains VGG-16 to generate latent codes.
"""
import sys
import os
import os.path
import argparse
import cv2
from glob import glob
from PIL import Image
import numpy as np
import time
from tqdm import tqdm
import copy
import re
import datetime

import matplotlib.pylab as plt
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, models, transforms

sys.path.append(os.path.join(os.getcwd(), "../"))

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger
from utils.manipulator import linear_interpolate

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Edit image synthesis with given semantic boundary.')
  parser.add_argument('-m', '--model_name', type=str, required=True,
                      choices=list(MODEL_POOL),
                      help='Name of the model for generation. (required)')
  parser.add_argument('-input_image', '--input_image', type=str, required=True,
                      help='Input image to find codes for')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('-v', '--pretrained_vgg_path', type=str,
                      help='Path to pretrained VGG-16 model')
  parser.add_argument('--epochs', type=int, default=100,
                      help='Number of steps for image editing. (default: 10)')
  parser.add_argument('--use_gpu', type=bool, default=True,
                      help='Whether or not to use the GPU.')

  return parser.parse_args()


class VGG9(nn.Module):
  def __init__(self, original_model):
    super(VGG9, self).__init__()
    #print("original model:")
    #print(original_model)
    layers = list(original_model.features.children())[:20]
    self.features = nn.Sequential(*layers)
    for param in list(self.features.parameters()):
      param.requires_grad = False
    #print("new model:")
    #print(self.features)

  def forward(self, x):
    #print(f"x is: {x}")
    y = self.features(x)
    #print(f"y is: {y}")
    return y[:,:,0,0]

def main():
  """Main function."""
  args = parse_args()
  logger = setup_logger(logger_name='latent_train')
  
  logger.info(f'Initializing generator.')
  gan_type = MODEL_POOL[args.model_name]['gan_type']
  if gan_type == 'pggan':
    model = PGGANGenerator(args.model_name, logger)
    kwargs = {}
  elif gan_type == 'stylegan':
    model = StyleGANGenerator(args.model_name, logger)
    kwargs = {'latent_space_type': args.latent_space_type}
  else:
    raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')

  logger.info(f'Preparing VGG.')
  stock_vgg = models.vgg16()
  if os.path.isfile(args.pretrained_vgg_path):
    logger.info(f'  Load vgg-16 state from `{args.pretrained_vgg_path}`.')
    stock_vgg.load_state_dict(torch.load(args.pretrained_vgg_path))
  else:
    raise NotImplementedError(f'  VGG16 initialized randomly. Is this really what you want?')
  vgg = VGG9(stock_vgg)

  if args.use_gpu:
    vgg.cuda()

  """
  R = your real image
  Gen(latent) - a generated image from some latent vector using pre-trained generator
  VGG16 - a pre-trained model for perceptual loss (9th layer in my implementation, but 5 also can be used)

  R_features = VGG16(R)
  G_features = VGG16(Gen(latent))

  loss = mse(R_features, G_features)
  ** only change latent**
  """

  gan_model = model # haha.
  vgg.train(False)
  vgg.eval()

  def normalize_image_to_arr(input_image):
    arr = np.array(input_image).astype(np.float32)

    # resize
    arr = cv2.resize(arr[:, :, ::-1], (224,224))

    # normalize for VGG
    arr[:,:,0] -= 103.939
    arr[:,:,1] -= 116.779
    arr[:,:,2] -= 123.68
    arr /= 255.0

    return arr
  
  def image_to_tensor(image):
    arr = normalize_image_to_arr(image)
    arr = arr.transpose((2, 0, 1)).astype(np.float32)
    tensor = torch.from_numpy(arr).float()
    tensor = tensor.unsqueeze(0)
    return tensor

  def image_from_gan(latent_code_tensor):
    latent_code = latent_code_tensor.cpu().detach().numpy()
    gan_outputs = gan_model.easy_synthesize(latent_code, **kwargs)
    gan_output_image = None
    for image in gan_outputs['image']:
      gan_output_image = image
      break
    return gan_output_image

  image = Image.open(args.input_image).resize((224,224))
  tensor_r = image_to_tensor(np.array(image))
  tensor_r = tensor_r.cuda()

  #numpy_codes = gan_model.easy_sample(1, **kwargs).astype(np.float32)
  #latent_tensor = torch.from_numpy(numpy_codes).float()
  #latent_code = Variable(latent_tensor.cuda(), requires_grad=True)

  weights = torch.randn(1, 512, device='cuda', requires_grad=True)
  criterion = nn.MSELoss()
  optimizer = optim.SGD([weights], lr=1, momentum=0.1)

  pbar = tqdm(range(args.epochs))
  for i in pbar:
    features_r = vgg(tensor_r)
    features_g = vgg(image_to_tensor(image_from_gan(weights)).cuda())

    left = features_g * weights
    right = features_r * weights

    loss = criterion(left, right)
    #print(f"loss was: {loss}")
    pbar.set_description(f"loss: {loss}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del features_g, features_r
    torch.cuda.empty_cache()
  np.save("codes.npy", weights.cpu().detach().numpy())

if __name__ == '__main__':
  import torch.multiprocessing
  torch.multiprocessing.set_start_method("spawn")
  main()
