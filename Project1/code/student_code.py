from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import random
import glob
import os
import numpy as np

import cv2
import numbers
import collections

import torch
from torch.utils import data

from utils import resize_image, load_image
from PIL import Image
# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################

class Compose(object):
  """Composes several transforms together.

  Args:
      transforms (list of ``Transform`` objects): list of transforms to compose.

  Example:
      >>> Compose([
      >>>     Scale(320),
      >>>     RandomSizedCrop(224),
      >>> ])
  """
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img):
    for t in self.transforms:
      img = t(img)
    return img

  def __repr__(self):
    repr_str = ""
    for t in self.transforms:
      repr_str += t.__repr__() + '\n'
    return repr_str

class RandomHorizontalFlip(object):
  """Horizontally flip the given numpy array randomly
     (with a probability of 0.5).
  """
  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be flipped.

    Returns:
        numpy array: Randomly flipped image
    """
    if random.random() < 0.5:
      img = cv2.flip(img, 1)
      return img
    return img

  def __repr__(self):
    return "Random Horizontal Flip"

#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
  """Rescale the input numpy array to the given size.

  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size, size * height / width)

      interpolations (list of int, optional): Desired interpolation.
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      Pass None during testing: always use CV2.INTER_LINEAR
  """
  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
    assert (isinstance(size, int)
            or (isinstance(size, collections.Iterable)
                and len(size) == 2)
           )
    self.size = size
    # use bilinear if interpolation is not specified
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be scaled.

    Returns:
        numpy array: Rescaled image
    """
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    # scale the image
    if isinstance(self.size, int):
      height, width, _ = img.shape
      if width < height:
        new_size = (self.size, int(self.size * height/width))
      else:
        new_size = (int(self.size * width/height), self.size)
      img = resize_image(img, new_size, interpolation)
      return img
    else:
      new_size = self.size
      img = resize_image(img, new_size, interpolation)    
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])

class RandomSizedCrop(object):
  """Crop the given numpy array to random area and aspect ratio.

  A crop of random area of the original size and a random aspect ratio
  of the original aspect ratio is made. This crop is finally resized to given size.
  This is widely used as data augmentation for training image classification models

  Args:
      size (sequence or int): size of target image. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          output size will be (size, size).
      interpolations (list of int, optional): Desired interpolation.
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      area_range (list of int): range of the areas to sample from
      ratio_range (list of int): range of aspect ratio to sample from
      num_trials (int): number of sampling trials
  """

  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS,
               area_range=(0.25, 1.0), ratio_range=(0.8, 1.2), num_trials=10):
    self.size = size
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations
    self.num_trials = int(num_trials)
    self.area_range = area_range
    self.ratio_range = ratio_range

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    for attempt in range(self.num_trials):

      # sample target area / aspect ratio from area range and ratio range
      area = img.shape[0] * img.shape[1]
      target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
      aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

      #################################################################################
      # Fill in the code here
      #################################################################################
      # compute the width and height
      # note that there are two possibilities
      # crop the image and resize to output size

      width = img.shape[1]
      height = img.shape[0]
      #print("width=%d, height=%d " %(width, height))

      x = random.randint(0, width-1)
      y = random.randint(0, height-1)

      #print("x=%d, y=%d" % (x, y))
      
      # aspect_ratio = width/height 
      target_height = int(math.sqrt(target_area/aspect_ratio))
      target_width = int(target_area/target_height)
      #print("target width=%d, target height=%d " %(target_width, target_height))
      
      if (y+target_height) <= height and (x+target_width) <= width:
        new_img = img[y:y+target_height, x:x+target_width, :]
        im_scale = Scale((self.size, self.size), interpolations=self.interpolations)      
        new_img = im_scale(new_img)
        return new_img

      # aspect_ratio = height/width
      target_width = int(math.sqrt(target_area/aspect_ratio))
      target_height = int(target_area/target_width)
      #print("target width=%d, target height=%d " %(target_width, target_height))
      
      if (y+target_height) <= height and (x+target_width) <= width:
        new_img = img[y:y+target_height, x:x+target_width, :]
        im_scale = Scale((self.size, self.size), interpolations=self.interpolations)      
        new_img = im_scale(new_img)
        return new_img

    # Fall back
    if isinstance(self.size, int):
      #print("fallback center")
      im_scale = Scale((self.size, self.size), interpolations=self.interpolations)
      img = im_scale(img)
      #################################################################################
      # Fill in the code here
      #################################################################################
      # with a square sized output, the default is to crop the patch in the center
      # (after all trials fail)
      #height, width = img.shape[0], img.shape[1]
      #x = width//2 - self.size//2
      #y = height//2 - self.size//2
      #img = img[y:y+height, x:x+width]
      return img
    else:
      # with a pre-specified output size, the default crop is the image itself
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img = im_scale(img)
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Random Crop" + \
           "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}%; Ratio {:.2f} - {:.2f}%]".format(
            target_size[0], target_size[1],
            self.area_range[0], self.area_range[1],
            self.ratio_range[0], self.ratio_range[1])


class RandomColor(object):
  """Perturb color channels of a given image
  Sample alpha in the range of (-r, r) and multiply 1 + alpha to a color channel.
  The sampling is done independently for each channel.

  Args:
      color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
  """
  def __init__(self, color_range):
    self.color_range = color_range

  def __call__(self, img):
    #################################################################################
    # Fill in the code here
    #################################################################################
    alpha = random.uniform(-self.color_range, self.color_range)
    img[:,:,0] = img[:,:,0] * (1 + alpha)

    alpha = random.uniform(-self.color_range, self.color_range)
    img[:,:,1] = img[:,:,1] * (1 + alpha)

    alpha = random.uniform(-self.color_range, self.color_range)
    img[:,:,2] = img[:,:,2] * (1 + alpha)
    # TODO: Prerformance Improvement
    return img

  def __repr__(self):
    return "Random Color [Range {:.2f} - {:.2f}%]".format(
            1-self.color_range, 1+self.color_range)


class RandomRotate(object):
  """Rotate the given numpy array (around the image center) by a random degree.

  Args:
      degree_range (float): range of degree (-d ~ +d)
  """
  def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
    self.degree_range = degree_range
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    # sample rotation
    degree = random.uniform(-self.degree_range, self.degree_range)
    # ignore small rotations
    if np.abs(degree) <= 1.0:
      return img

    #################################################################################
    # Fill in the code here
    #################################################################################
    # get the max rectangular within the rotated image
    pil_img = Image.fromarray(img)
    pil_img = pil_img.rotate(degree)
    img = np.array(pil_img)

    radians = math.radians(degree)
    quadrant = int(math.floor(radians/(math.pi/2))) & 3
    if (quadrant & 1) == 0:
      sign_alpha = radians
    else:
      sign_alpha = math.pi - radians
    alpha = ((sign_alpha % math.pi) + math.pi) % math.pi
    bb = {
        "width": img.shape[1] * math.cos(alpha) + img.shape[0] * math.sin(alpha),
        "height": img.shape[1]* math.sin(alpha) + img.shape[0] * math.cos(alpha)
    }
    if img.shape[1] < img.shape[0]:
      gamma = math.atan2(bb["width"], bb["height"])
    else:
      gamma = math.atan2(bb["height"], bb["width"])

    delta = math.pi - alpha - gamma

    if img.shape[1] < img.shape[0] :
     length = img.shape[0]
    else:
      length = img.shape[1]
    d = length * math.cos(alpha);
    a = d * math.sin(alpha) / math.sin(delta);

    y = int(a * math.cos(gamma));
    x = int(y * math.tan(gamma));
    #print(y, (bb["height"] - 2 * y), x, (bb["width"] - 2 * x))
    new_img = img[y : int((bb["height"] - 2 * y)), x : int((bb["width"] - 2 * x))]
    im_scale = Scale((img.shape[1], img.shape[0]), interpolations=self.interpolations)
    new_img = im_scale(new_img)
    return new_img

  def __repr__(self):
    return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range)

#################################################################################
# Additional helper functions
#################################################################################
class ToTensor(object):
  """Convert a ``numpy.ndarray`` image to tensor.
  Converts a numpy.ndarray (H x W x C) image in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """
  def __call__(self, img):
    assert isinstance(img, np.ndarray)
    # convert image to tensor
    assert (img.ndim > 1) and (img.ndim <= 3)
    if img.ndim == 2:
      img = img[:, :, None]
      tensor_img = torch.from_numpy(np.ascontiguousarray(
        img.transpose((2, 0, 1))))
    if img.ndim == 3:
      tensor_img = torch.from_numpy(np.ascontiguousarray(
        img.transpose((2, 0, 1))))
    # backward compatibility
    if isinstance(tensor_img, torch.ByteTensor):
      return tensor_img.float().div(255.0)
    else:
      return tensor_img

class SimpleDataset(data.Dataset):
  """
  A simple dataset
  """
  def __init__(self, root_folder, file_ext, transforms=None):
    # root folder, split
    self.root_folder = root_folder
    self.transforms = transforms
    self.file_ext = file_ext

    # load all labels
    file_list = glob.glob(os.path.join(root_folder, '*.{:s}'.format(file_ext)))
    self.file_list = file_list

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, index):
    # load img and label (from file name)
    filename = self.file_list[index]
    img = load_image(filename)
    label = os.path.basename(filename)
    label = label.rstrip('.{:s}'.format(self.file_ext))
    # apply data augmentation
    if self.transforms is not None:
      img  = self.transforms(img)
    return img, label
