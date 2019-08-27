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

import vgg_face_dag

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
  parser.add_argument('-train_dir', '--train_dir', type=str, required=True,
                      help='Training directory (required)')
  parser.add_argument('-test_dir', '--test_dir', type=str, required=True,
                      help='Test directory (required)')
  parser.add_argument('-val_dir', '--val_dir', type=str, required=True,
                      help='Validation directory (required)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('-v', '--pretrained_vgg_path', type=str,
                      help='Path to pretrained VGG-16 model')
  parser.add_argument('--epochs', type=int, default=100,
                      help='Number of steps for image editing. (default: 10)')
  parser.add_argument('--batch_size', type=int, default=4,
                      help='Number of latent codes per batch (default: 10)')
  parser.add_argument('--use_gpu', type=bool, default=True,
                      help='Whether or not to use the GPU.')
  parser.add_argument('--resume', type=bool, default=False,
                      help='Whether or not to resume from the latest checkpoint.')

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

class ImageDataset(Dataset):
  def __init__(self, directory, transform=None):
    self.dir = directory
    self.jpgs = self.load_into_dict(glob(os.path.join(directory, "*.jpg")))
    self.npys = self.load_into_dict(glob(os.path.join(directory, "*.npy")))
    assert len(self.jpgs) == len(self.npys)
    self.length = len(self.jpgs)
    self.transform = transform
    
  def load_into_dict(self, files):
    m = re.compile(r"(\d+)", re.X)
    dictionary = {}
    for f in files:
      index = int(m.findall(f)[0])
      dictionary[index] = f
    return dictionary

  def __getitem__(self, index):
    if index > self.length:
      return NotImplementedError(f"seeking out of bounds in GanDatasetLoader length: {self.length}")
    latent_codes = np.load(self.npys[index])
    image = Image.open(self.jpgs[index])
    arr = np.array(image, dtype=np.float32)
    arr[:,:,0] -= 103.939
    arr[:,:,1] -= 116.779
    arr[:,:,2] -= 123.68
    arr /= 255.
    image = Image.fromarray(arr.astype('uint8'))
    if self.transform is not None:
      image = self.transform(image)
    return (image, latent_codes.squeeze())

  def __len__(self):
    return len(self.jpgs)

def eval_model(vgg, dataloaders, dataset_sizes, criterion, use_gpu=True):
    since = time.time()
    avg_loss = 0
    loss_test = 0
    
    test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(tqdm(dataloaders[TEST])):
        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        with torch.no_grad():
          if use_gpu:
              inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
          else:
              inputs, labels = Variable(inputs), Variable(labels)

          inputs = inputs.to('cuda')
          outputs = vgg(inputs)
          loss = criterion(outputs, labels)
          loss_test += loss

          del inputs, labels, outputs
          torch.cuda.empty_cache()
        
    avg_loss = loss_test / dataset_sizes[TEST]
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print('-' * 10)

def train_model(vgg, gan_model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10, use_gpu=True, gan_kwargs=None, start_epoch=0):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_loss = 1e6
    avg_loss = 0
    avg_loss_val = 0
    
    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[VAL])
    
    for epoch in range(start_epoch + 1, start_epoch + 1 + num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_val = 0
        
        vgg.train(True)
        print(f"Training... epoch {epoch}")
        for i, data in enumerate(tqdm(dataloaders[TRAIN])):
            if i > 1000: break
            inputs, labels = data            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            inputs = inputs.to('cuda')            
            outputs = vgg(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()            
            loss_train += loss
            
            del inputs, labels, outputs
            torch.cuda.empty_cache()
        
        avg_loss = loss_train / dataset_sizes[TRAIN]
        
        vgg.train(False)
        vgg.eval()

        print(f"Evaluating... epoch: {epoch}")
        for i, data in enumerate(tqdm(dataloaders[VAL])):
            inputs, labels = data            
            with torch.no_grad():
              if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
              else:
                inputs, labels = Variable(inputs), Variable(labels)

              optimizer.zero_grad()
              inputs = inputs.to('cuda')
              outputs = vgg(inputs)
              loss = criterion(outputs, labels)
              loss_val += loss

              del inputs, labels, outputs
              torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / dataset_sizes[VAL]
        
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print('-' * 10)
        print()

        print(f"Visualizing model results at epoch: {epoch}")
        visualize_model(vgg, gan_model, epoch, num_examples=3, **gan_kwargs)

        # only save or visualize every 5 epochs.
        if epoch % 5 == 0:
          if avg_loss_val < best_loss:
            print(f"updating best model weights (loss {avg_loss_val} < {best_loss})")
            best_loss = avg_loss_val
            best_model_wts = copy.deepcopy(vgg.state_dict())
            ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            outfile = f"./checkpoints/latent_VGG16_{ts}_{epoch}.pt"
            print(f"saving checkpoint to: {outfile}")
            torch.save(best_model_wts, outfile)

    elapsed_time = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best loss: {:.4f}".format(best_loss))

    vgg.load_state_dict(best_model_wts)
    return vgg

def visualize_model(vgg_model, gan_model, epoch, num_examples=3, **kwargs):
  vgg_model.train(False)
  vgg_model.eval()
  latent_codes = gan_model.easy_sample(num_examples, **kwargs)
  outputs = gan_model.easy_synthesize(latent_codes, **kwargs)
  gan_output_images = []
  for image in outputs['image']:
    # transform image to 224x224 for rest of uses
    resized = cv2.resize(image[:, :, ::-1], (224,224))
    gan_output_images.append(resized)

  my_latent_codes = []
  for image in gan_output_images:
    tensor = toTensor(image).unsqueeze(0)
    tensor = tensor.to('cuda')
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
    print("original model:")
    print(original_model)
    layers = list(original_model.features.children())[:28]
    self.features = nn.Sequential(*layers)
#    for param in list(self.features.parameters())[:-4]:
#      param.requires_grad = False
    print("new model:")
    print(self.features)

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

  data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally. 
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
   }

  image_datasets = {
    TRAIN: ImageDataset(args.train_dir,
      transform=data_transforms[TRAIN],
    ),
    VAL: ImageDataset(args.val_dir,
      transform=data_transforms[VAL],
    ),
    TEST:ImageDataset(args.test_dir,
      transform=data_transforms[TEST],
    ),
  }

  dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=args.batch_size,
        shuffle=True
    )
    for x in [TRAIN, VAL, TEST]
  }

  dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

  logger.info(f'Preparing VGG.')
  # Load the pretrained model from pytorch
  #stock_vgg = models.vgg16()
  stock_vgg = models.vgg16_bn()
  #stock_vgg = vgg_face_dag.Vgg_face_dag()
  if os.path.isfile(args.pretrained_vgg_path):
    logger.info(f'  Load vgg-16 state from `{args.pretrained_vgg_path}`.')
    stock_vgg.load_state_dict(torch.load(args.pretrained_vgg_path))
  else:
    raise NotImplementedError(f'  VGG16 initialized randomly. Is this really what you want?')
  vgg = VGG9(stock_vgg)

  start_epoch = 0
  if args.resume:
    start_epoch, latest = find_latest_epoch_and_checkpoint("./checkpoints/")
    if latest:
      print(f"restoring from checkpoint: {latest} (epoch: {start_epoch})")
      vgg.load_state_dict(torch.load(latest))

  params_to_update = []
  for name, param in vgg.named_parameters():
    if param.requires_grad == True:
      params_to_update.append(param)
      print("\t",name)

  if args.use_gpu:
    vgg.cuda()

  criterion = nn.MSELoss()
  #optimizer_ft = optim.Adam(params_to_update, lr=0.1, betas=[0.5, 0.999])
  optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
  print("Visualizing model...\n")
  visualize_model(vgg, model, 999, num_examples=4, **kwargs)

  print("Test before training...\n")
  eval_model(vgg, dataloaders, dataset_sizes, criterion, use_gpu=args.use_gpu)

  print("Training model...\n")
  vgg = train_model(vgg, model, dataloaders, dataset_sizes, criterion, optimizer_ft,
                      num_epochs=args.epochs, use_gpu=args.use_gpu, gan_kwargs=kwargs, start_epoch=start_epoch)
  print("Test after training...\n")  
  eval_model(vgg, dataloaders, dataset_sizes, criterion, use_gpu=args.use_gpu)


if __name__ == '__main__':
  import torch.multiprocessing
  torch.multiprocessing.set_start_method("spawn")
  main()
