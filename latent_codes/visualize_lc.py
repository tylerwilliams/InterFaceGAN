# python3.7
"""Retrains VGG-16 to generate latent codes.
"""
import os.path
import os
import sys
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

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Edit image synthesis with given semantic boundary.')
  parser.add_argument('-m', '--model_name', type=str, required=True,
                      choices=list(MODEL_POOL),
                      help='Name of the model for generation. (required)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('--use_gpu', type=bool, default=True,
                      help='Whether or not to use the GPU.')
  parser.add_argument('--codes_file', type=str, default=None,
                      help='Latent codes file to visualize.')

  return parser.parse_args()

def toTensor(img):
  """convert a numpy array of shape HWC to CHW tensor"""
  img = img.transpose((2, 0, 1)).astype(np.float32)
  tensor = torch.from_numpy(img).float()
  return tensor/255.0

def find_latest_epoch_and_checkpoint(checkpoints_dir):
  # files like: latent_VGG16_20190824182850_46.pt
  checkpoints = sorted(glob(f"{checkpoints_dir}/*.pt"))
  if len(checkpoints) == 0:
    return 0, None
  matcher = re.compile(r".*_(\d+)_(\d+).pt", re.X)
  latest = checkpoints[-1]
  _, epoch = matcher.findall(os.path.basename(latest))[0]
  return int(epoch), latest

def visualize_latent_codes(gan_model, codes_file, **kwargs):
  latent_codes = np.load(codes_file)
  outputs = gan_model.easy_synthesize(latent_codes, **kwargs)
  gan_output_image = None
  for image in outputs['image']:
    gan_output_image = image
    break
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.imshow(gan_output_image)
  ax.title.set_text('GAN image')
  fig.savefig(f'model_viz_fixed_codes.png')
  del fig

def visualize_model(vgg_model, gan_model, epoch, num_examples=3, **kwargs):
  vgg_model.train(False)
  vgg_model.eval()
  latent_codes = gan_model.easy_sample(num_examples, **kwargs)
  outputs = gan_model.easy_synthesize(latent_codes, **kwargs)
  gan_output_images = []
  for image in outputs['image']:
    # transform image to 224x224 for rest of uses
    print("image type: ", type(image))
    resized = cv2.resize(image[:, :, ::-1], (224,224))
    gan_output_images.append(resized)

  my_latent_codes = []
  for image in gan_output_images:
    print("image shape: ",image.shape)
    tensor = toTensor(image).unsqueeze(0)
    tensor = tensor.to('cuda')
    print("tensor shape:",tensor.shape)
    image_latent_codes = vgg_model(tensor)
    my_latent_codes.append(image_latent_codes.cpu().detach().numpy())

  # import pdb; pdb.set_trace()
  secondary_gan_output_images = []
  secondary_outputs = gan_model.easy_synthesize(np.vstack(my_latent_codes), **kwargs)
  for image in secondary_outputs['image']:
    resized = cv2.resize(image[:, :, ::-1], (224,224))
    secondary_gan_output_images.append(resized)

  fig = plt.figure()
  for i, (gan_img, secondary_img) in enumerate(zip(gan_output_images, secondary_gan_output_images)):
    ax = fig.add_subplot(num_examples, 2, i*2 + 1)
    ax.imshow(cv2.cvtColor(gan_img, cv2.COLOR_BGR2RGB))
    ax.title.set_text('GAN image')

    ax2 = fig.add_subplot(num_examples, 2, i*2 + 2)
    ax2.imshow(cv2.cvtColor(secondary_img, cv2.COLOR_BGR2RGB))
    ax2.title.set_text('Latent-codes image')
  fig.savefig(f'model_viz_epoch_{epoch}.png')
  del fig

class VGG9(nn.Module):
  def __init__(self, original_model):
    super(VGG9, self).__init__()
    layers = list(original_model.features.children())[:28]
    self.features = nn.Sequential(*layers)

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
  # Load the pretrained model from pytorch
  #stock_vgg = models.vgg16()
  stock_vgg = models.vgg16_bn()
  #stock_vgg = vgg_face_dag.Vgg_face_dag()
  vgg = VGG9(stock_vgg)

  start_epoch = 0
  start_epoch, latest = find_latest_epoch_and_checkpoint("./checkpoints/")
  if latest:
    print(f"restoring from checkpoint: {latest} (epoch: {start_epoch})")
    vgg.load_state_dict(torch.load(latest))
  else:
    print("No checkpoint found! exiting")
    return 1

  if args.use_gpu:
    vgg.cuda()

  if args.codes_file:
    visualize_latent_codes(model, args.codes_file, **kwargs)
  else:
    print("Visualizing model...\n")
    visualize_model(vgg, model, 999, num_examples=4, **kwargs)

if __name__ == '__main__':
  import torch.multiprocessing
  torch.multiprocessing.set_start_method("spawn")
  main()
