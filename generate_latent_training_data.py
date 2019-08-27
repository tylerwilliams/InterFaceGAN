# python3.7
"""Retrains VGG-16 to generate latent codes.
"""

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
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('--num_train', type=int, default=0,
                      help='Number of training images to write')
  parser.add_argument('--num_test', type=int, default=0,
                      help='Number of test images to write')
  parser.add_argument('--num_val', type=int, default=0,
                      help='Number of validation images to write')

  return parser.parse_args()

def write_images(model, num_to_write, output_dir_path, gan_kwargs=dict()):
  for i in tqdm(range(num_to_write)):
    latent_codes = model.easy_sample(1, **gan_kwargs)
    outputs = model.easy_synthesize(latent_codes, **gan_kwargs)
    gan_output_images = []
    image =  outputs['image'][0]
    # transform image to 224x224 for rest of uses
    image = cv2.resize(image[:, :, ::-1], (224,224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.save(os.path.join(output_dir_path, f"fake_{i}.jpg"))
    np.save(os.path.join(output_dir_path, f"fake_{i}_codes.npy"), latent_codes)

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

  write_images(model, args.num_train, "./train", gan_kwargs=kwargs)
  write_images(model, args.num_test, "./test", gan_kwargs=kwargs)
  write_images(model, args.num_val, "./validation", gan_kwargs=kwargs)

if __name__ == '__main__':
  import torch.multiprocessing
  torch.multiprocessing.set_start_method("spawn")
  main()
