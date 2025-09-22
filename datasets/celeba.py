# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import hashlib
import logging
import os
import subprocess
import zipfile

import numpy as np
from PIL import Image
import skimage.io
import torch
import torchvision

import datasets

class CelebA(torch.utils.data.Dataset):
    """CelebA Dataset from [1].
    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.
    
    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    - This dataset doesn't have controlled generative factors of variation like synthetic datasets,
      so it inherits directly from torch.utils.data.Dataset instead of DisentangledDataset.
    
    Parameters
    ----------
    root : string
        Root directory of dataset.
        
    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).
    """
    urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    files = {"train": "img_align_celeba"}
    img_size = (3, 64, 64)
    background_color = datasets.COLOUR_WHITE

    def __init__(self, root='data/celeba', transforms=None, subset=1.0, logger=None, **kwargs):
        """Initialize the CelebA dataset.
        
        Parameters
        ----------
        root : str, default='data/celeba'
            Root directory where the dataset will be downloaded and stored.
        transforms : torchvision.transforms.Compose or list, optional
            Transforms to apply to the images. If None, defaults to ToTensor().
        subset : float, default=1.0
            Fraction of the dataset to use (between 0 and 1).
        logger : logging.Logger, optional
            Logger instance. If None, creates a default logger.
        **kwargs : 
            Additional arguments for compatibility.
        """
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.subset = subset
        self.logger = logger or logging.getLogger(__name__)
        
        # Set up transforms
        if transforms is None:
            self.transforms = torchvision.transforms.ToTensor()
        elif isinstance(transforms, list):
            self.transforms = torchvision.transforms.Compose(transforms)
        else:
            self.transforms = transforms

        # Download dataset if it doesn't exist
        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

        # Load image paths
        self.img_paths = sorted(glob.glob(os.path.join(self.train_data, '*')))
        
        # Apply subset if specified
        if self.subset < 1.0:
            num_samples = int(len(self.img_paths) * self.subset)
            self.img_paths = self.img_paths[:num_samples]

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'celeba.zip')
        os.makedirs(self.root, exist_ok=True)
        
        self.logger.info("Downloading CelebA dataset...")
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", save_path])

        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        with open(save_path, 'rb') as f:
            actual_hash = hashlib.md5(f.read()).hexdigest()
        assert actual_hash == hash_code, \
            '{} file is corrupted.  Remove the file and try again.'.format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.logger.info("Extracting CelebA ...")
            zf.extractall(self.root)

        os.remove(save_path)

        self.logger.info("Resizing CelebA ...")
        self._preprocess_images()

    def _preprocess_images(self):
        """Preprocess images to the target size."""
        img_dir = self.train_data
        img_paths = glob.glob(os.path.join(img_dir, '*'))
        
        target_size = type(self).img_size[1:]  # (H, W)
        
        for img_path in img_paths:
            # Load image
            img = Image.open(img_path)
            
            # Resize image
            img_resized = img.resize(target_size, Image.LANCZOS)
            
            # Save back
            img_resized.save(img_path)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Get the image at index `idx`.
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.
            
        Returns
        -------
        img : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        placeholder : int
            Placeholder value (0) since there are no labels.
        """
        img_path = self.img_paths[idx]
        
        # Load and transform image
        img = skimage.io.imread(img_path)
        img = self.transforms(img)
        
        # Return image with placeholder label (0) for compatibility
        return img, 0