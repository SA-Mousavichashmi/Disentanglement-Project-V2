# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
import subprocess
import zipfile

import numpy as np
from PIL import Image
import skimage.io
import torch
import torchvision

try:
    import gdown
except ImportError:
    gdown = None

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
    urls = {
        "train": "https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ",
        "landmarks": "https://drive.google.com/uc?id=0B7EVK8r0v71pd0FJY3Blby1HUTQ"  # Face landmarks
    }
    files = {"train": "img_align_celeba.zip", "landmarks": "list_landmarks_align_celeba.txt"}
    img_size = (3, 64, 64)
    background_color = datasets.COLOUR_WHITE

    def __init__(self, root='data/celeba', transforms=None, subset=1.0, logger=None, 
                 resize_algorithm='LANCZOS', crop_faces=False, download_annotations=True, 
                 crop_margin=0.3, force_download=False, **kwargs):
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
        resize_algorithm : str, default='LANCZOS'
            Image resizing algorithm. Options: 'LANCZOS', 'BICUBIC'.
        crop_faces : bool, default=False
            Whether to crop images to face regions based on landmarks.
        download_annotations : bool, default=True
            Whether to download face landmarks (required for face cropping).
        crop_margin : float, default=0.3
            Margin factor for face cropping to include head and hair (0.3 = 30% margin).
        force_download : bool, default=False
            Whether to force redownload and reprocess the dataset, useful when changing flags like crop_faces.
        **kwargs : 
            Additional arguments for compatibility.
        """
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.subset = subset
        self.logger = logger or logging.getLogger(__name__)
        self.resize_algorithm = resize_algorithm.upper()
        self.crop_faces = crop_faces
        self.download_annotations = download_annotations
        self.crop_margin = crop_margin
        self.face_landmarks = {}  # Store face landmarks
        
        # Validate resize algorithm
        if self.resize_algorithm not in ['LANCZOS', 'BICUBIC']:
            raise ValueError("resize_algorithm must be 'LANCZOS' or 'BICUBIC'")
        
        # Validate crop margin
        if not (0.0 <= crop_margin <= 1.0):
            raise ValueError("crop_margin must be between 0.0 and 1.0")
        
        # Set up transforms
        if transforms is None:
            self.transforms = torchvision.transforms.ToTensor()
        elif isinstance(transforms, list):
            self.transforms = torchvision.transforms.Compose(transforms)
        else:
            self.transforms = transforms

        # Check if dataset exists; no automatic download
        if not os.path.isdir(self.train_data):
            raise FileNotFoundError(
                f"CelebA dataset not found at {self.train_data}. "
                "Please download manually:\n"
                "1. Main images: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view "
                "(save as data/celeba/celeba.zip and extract to data/celeba/img_align_celeba/)\n"
                "2. For face cropping: https://drive.google.com/uc?id=0B7EVK8r0v71pd0FJY3Blby1HUTQ "
                "(save as data/celeba/list_landmarks_align_celeba.txt)\n"
                "Then re-initialize the dataset."
            )

        # Load image paths
        self.img_paths = sorted(glob.glob(os.path.join(self.train_data, '*')))
        
        # Load face annotations if needed
        if self.crop_faces and self.download_annotations:
            self._load_face_annotations()
        
        # Apply subset if specified
        if self.subset < 1.0:
            num_samples = int(len(self.img_paths) * self.subset)
            self.img_paths = self.img_paths[:num_samples]

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'celeba.zip')
        os.makedirs(self.root, exist_ok=True)
        
        # Check if zip file already exists and is valid
        valid_zip = False
        if os.path.exists(save_path):
            try:
                with zipfile.ZipFile(save_path, 'r') as zf:
                    zf.testzip()  # Test if zip file is valid
                valid_zip = True
                self.logger.info("Valid CelebA zip file already exists, skipping download.")
            except (zipfile.BadZipFile, zipfile.LargeZipFile):
                self.logger.warning("Existing zip file is corrupted, will re-download.")
                os.remove(save_path)
        
        if not valid_zip:
            if gdown is None:
                raise ImportError("gdown is required for downloading CelebA dataset. Install it with: pip install gdown")
            
            self.logger.info("Downloading CelebA dataset...")
            # Use direct download URL for large files
            url = "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
            try:
                success = gdown.download(url=url, output=save_path, quiet=False, confirm=True)
                if not success:
                    raise RuntimeError("gdown.download returned False - download failed")
                
                # Verify the downloaded file is a valid zip
                with zipfile.ZipFile(save_path, 'r') as zf:
                    zf.testzip()
                self.logger.info("Download successful and zip file validated.")
                
            except Exception as e:
                self.logger.error(f"Failed to download CelebA dataset: {e}")
                # Clean up corrupted file
                if os.path.exists(save_path):
                    os.remove(save_path)
                raise

        # Extract the zip file
        try:
            with zipfile.ZipFile(save_path) as zf:
                self.logger.info("Extracting CelebA ...")
                zf.extractall(self.root)
        except zipfile.BadZipFile as e:
            self.logger.error(f"Downloaded file is not a valid zip file: {e}")
            # Clean up corrupted file and raise error
            if os.path.exists(save_path):
                os.remove(save_path)
            raise

        # Keep the zip file for future use; do not delete it
        # os.remove(save_path)

        # Download face annotations if required
        if self.download_annotations or self.crop_faces:
            self.logger.info("Downloading face landmarks...")
            self._download_annotations()

        self.logger.info("Resizing CelebA ...")
        self._preprocess_images()

    def _download_annotations(self):
        """Download face landmark annotations."""
        landmarks_path = os.path.join(self.root, type(self).files["landmarks"])
        
        if gdown is None:
            self.logger.error("gdown is required for downloading face landmarks. Install it with: pip install gdown")
            self.crop_faces = False
            return
        
        # Check if landmarks file already exists and is valid
        if os.path.exists(landmarks_path):
            try:
                # Try to read a few lines to validate the file
                with open(landmarks_path, 'r') as f:
                    lines = f.readlines()[:5]
                    if len(lines) > 0 and not all(line.strip().startswith('<') for line in lines):
                        self.logger.info("Valid landmarks file already exists, skipping download.")
                        return
            except Exception:
                self.logger.warning("Existing landmarks file is corrupted, will re-download.")
                os.remove(landmarks_path)
        
        # Use direct download URL
        url = "https://drive.google.com/uc?id=0B7EVK8r0v71pd0FJY3Blby1HUTQ"
        
        self.logger.info("Downloading face landmarks with gdown...")
        try:
            success = gdown.download(url=url, output=landmarks_path, quiet=False, confirm=True)
            if not success:
                raise RuntimeError("gdown.download returned False - download failed")
                
            # Validate the downloaded file
            with open(landmarks_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('<') or 'error' in first_line.lower():
                    raise RuntimeError("Downloaded file appears to be an error page, not the landmarks file")
                    
            self.logger.info("Landmarks download successful and file validated.")
            
        except Exception as e:
            self.logger.error(f"Failed to download landmarks: {e}. Face cropping will be disabled.")
            self.crop_faces = False
            # Clean up corrupted file
            if os.path.exists(landmarks_path):
                os.remove(landmarks_path)
            return
        
        if not os.path.exists(landmarks_path):
            self.logger.error("Landmarks file not found after download. Face cropping will be disabled.")
            self.crop_faces = False

    def _load_face_annotations(self):
        """Load and parse face landmark annotations."""
        landmarks_path = os.path.join(self.root, type(self).files["landmarks"])
        
        if not os.path.exists(landmarks_path):
            self.logger.warning("Face landmarks file not found. Face cropping will be disabled.")
            self.crop_faces = False
            return
            
        self.logger.info("Loading face landmarks...")
        
        with open(landmarks_path, 'r') as f:
            lines = f.readlines()
            
        # Parse landmarks - format: image_name lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
        for line in lines:
            line = line.strip()
            if not line or line.startswith('lefteye_x'):  # Skip header line if present
                continue
                
            parts = line.split()
            if len(parts) == 11:  # image_name + 10 coordinates (5 landmarks * 2 coordinates each)
                img_name = parts[0]
                # Parse landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
                landmarks = []
                for i in range(1, 11, 2):  # Step by 2 to get (x, y) pairs
                    x, y = int(parts[i]), int(parts[i + 1])
                    landmarks.append((x, y))
                self.face_landmarks[img_name] = landmarks
        
        self.logger.info(f"Loaded {len(self.face_landmarks)} face landmark annotations.")

    def _compute_face_bbox_from_landmarks(self, landmarks):
        """Compute face bounding box from landmarks with margin for head and hair.
        
        Parameters
        ----------
        landmarks : list of tuples
            List of 5 (x, y) landmark coordinates: 
            [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
            
        Returns
        -------
        tuple
            (x_min, y_min, x_max, y_max) bounding box coordinates
        """
        if len(landmarks) != 5:
            return None
            
        # Extract coordinates
        xs = [point[0] for point in landmarks]
        ys = [point[1] for point in landmarks]
        
        # Get face region bounds
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Calculate face dimensions
        face_width = x_max - x_min
        face_height = y_max - y_min
        
        # Add margin to include head and hair
        margin_x = face_width * self.crop_margin
        margin_y = face_height * self.crop_margin
        
        # Expand bounding box with margins
        x_min = max(0, int(x_min - margin_x))
        y_min = max(0, int(y_min - margin_y * 1.5))  # More margin on top for hair
        x_max = int(x_max + margin_x)
        y_max = int(y_max + margin_y * 0.5)  # Less margin on bottom
        
        return (x_min, y_min, x_max, y_max)

    def _preprocess_images(self):
        """Preprocess images to the target size."""
        img_dir = self.train_data
        img_paths = glob.glob(os.path.join(img_dir, '*'))
        
        target_size = type(self).img_size[1:]  # (H, W)
        
        # Get resize algorithm
        resize_method = getattr(Image, self.resize_algorithm)
        
        for img_path in img_paths:
            # Load image
            img = Image.open(img_path)
            original_size = img.size  # (width, height)
            
            # Apply face cropping if enabled
            if self.crop_faces:
                img_name = os.path.basename(img_path)
                if img_name in self.face_landmarks:
                    landmarks = self.face_landmarks[img_name]
                    bbox = self._compute_face_bbox_from_landmarks(landmarks)
                    if bbox:
                        x_min, y_min, x_max, y_max = bbox
                        # Ensure bbox is within image bounds
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(original_size[0], x_max)
                        y_max = min(original_size[1], y_max)
                        
                        # Crop to face region with margin
                        img = img.crop((x_min, y_min, x_max, y_max))
                    else:
                        self.logger.warning(f"Could not compute face bbox for {img_name}, using full image.")
                else:
                    self.logger.warning(f"No face landmarks found for {img_name}, using full image.")
            
            # Resize image with configurable algorithm
            img_resized = img.resize(target_size, resize_method)
            
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