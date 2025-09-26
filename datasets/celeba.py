# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
import shutil
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
        "train": "https://drive.google.com/uc?id=16iSaRTazPPEEs3y68pQPSeXjOiJ9o7VX",
        "annotations": "https://drive.google.com/file/d/1doqk-enfMbxU3T_KZ1UVIxA58rBZAPyV/view?usp=drive_link"  # CelebA annotations
    }
    files = {"train": "img_align_celeba.zip", "annotations": "celeba_annotations.zip"}
    img_size = (3, 64, 64)
    background_color = datasets.COLOUR_WHITE

    def __init__(self, root='data/celeba', transforms=None, subset=1.0, logger=None, 
                 resize_algorithm='LANCZOS', crop_faces=False, 
                 crop_margin=0.6, force_download=False, load_into_memory=True, **kwargs):
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
        crop_margin : float, default=0.3
            Margin factor for face cropping to include head and hair (0.3 = 30% margin).
        force_download : bool, default=False
            Whether to force redownload and reprocess the dataset, useful when changing flags like crop_faces.
        load_into_memory : bool, default=False
            Load all images into RAM during initialization for faster subsequent access.
        **kwargs : 
            Additional arguments for compatibility.
        """
        self.root = root
        self.subset = subset
        self.logger = logger or logging.getLogger(__name__)
        self.resize_algorithm = resize_algorithm.upper()
        self.crop_faces = crop_faces
        self.crop_margin = crop_margin
        self.force_download = force_download
        self.load_into_memory = load_into_memory
        self._img_cache = None
        self.face_landmarks = {}  # Store face landmarks
        self.raw_data_dir = os.path.join(self.root, 'img_align_celeba')
        self.processed_data_dir = os.path.join(self.root, 'img_align_celeba_preprocssed')
        self.annotations_dir = os.path.join(self.root, 'celeba_annotations')
        
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

        needs_download = self.force_download or not os.path.exists(self.raw_data_dir)
        if self.crop_faces:
            needs_download = needs_download or not os.path.exists(self.annotations_dir)

        if needs_download:
            self.logger.info("Dataset not found or force download requested, downloading...")
            self.download()

        self.train_data = self.raw_data_dir

        if self.crop_faces:
            processed_exists = os.path.exists(self.processed_data_dir)
            if processed_exists:
                self.logger.info("Face-preprocessed images directory already exists, skipping preprocessing.")
                self.train_data = self.processed_data_dir
            else:
                self._load_face_annotations()
                if not self.face_landmarks:
                    self.logger.warning("No face landmarks available, cannot perform face cropping.")
                    self.crop_faces = False
                else:
                    self.logger.info("Starting face preprocessing...")
                    self._preprocess_images(self.raw_data_dir, self.processed_data_dir)
                    if os.path.exists(self.processed_data_dir):
                        self.logger.info("Face preprocessing completed.")
                        self.train_data = self.processed_data_dir
                    else:
                        self.logger.warning("Face preprocessing failed to create processed images directory, using original images.")
                        self.crop_faces = False

        # Load image paths
        self.img_paths = sorted(glob.glob(os.path.join(self.train_data, '*.jpg')))

        if not self.img_paths and self.train_data == self.processed_data_dir and os.path.exists(self.raw_data_dir):
            self.logger.warning("Preprocessed directory is empty, falling back to original images.")
            self.crop_faces = False
            self.train_data = self.raw_data_dir
            self.img_paths = sorted(glob.glob(os.path.join(self.train_data, '*.jpg')))
        
        # Apply subset if specified
        if self.subset < 1.0:
            num_samples = int(len(self.img_paths) * self.subset)
            self.img_paths = self.img_paths[:num_samples]

        if self.load_into_memory:
            if not self.img_paths:
                self.logger.warning("No images found to load into memory.")
            else:
                self._img_cache = [None] * len(self.img_paths)
                self._load_images_into_memory()

    @property
    def name(self):
        """Name of the dataset."""
        return 'celeba'

    @property
    def kwargs(self):
        """Keyword arguments for the dataset."""
        return {
            'root': self.root,
            'subset': self.subset,
            'resize_algorithm': self.resize_algorithm,
            'crop_faces': self.crop_faces,
            'crop_margin': self.crop_margin,
            'force_download': self.force_download,
            'load_into_memory': self.load_into_memory
        }
    
    def download(self):
        """Download the dataset."""
        if gdown is None:
            raise ImportError("gdown is required for downloading CelebA dataset. Install it with: pip install gdown")
        
        # Determine which components need downloading
        img_dir = self.raw_data_dir
        ann_dir = self.annotations_dir

        download_images = self.force_download or not os.path.exists(img_dir)
        download_annotations = False
        if self.crop_faces:
            download_annotations = self.force_download or not os.path.exists(ann_dir)

        if not download_images and not download_annotations:
            self.logger.info("Required dataset directories already exist, skipping download.")
            return
        
        # Create directory structure
        os.makedirs(self.root, exist_ok=True)
        original_zip_dir = os.path.join(self.root, 'original_zip_files')
        os.makedirs(original_zip_dir, exist_ok=True)
        
        # Download and extract images if needed
        if download_images:
            if self.force_download and os.path.exists(img_dir):
                shutil.rmtree(img_dir)
            img_zip_path = os.path.join(original_zip_dir, self.files["train"])
            if not self._download_file("train", img_zip_path):
                return
            self._extract_zip(img_zip_path, self.root, 'img_align_celeba')
        else:
            self.logger.info("Image directory already present, skipping image download.")

        # Download and extract annotations if needed
        if download_annotations:
            if self.force_download and os.path.exists(ann_dir):
                shutil.rmtree(ann_dir)
            ann_zip_path = os.path.join(original_zip_dir, self.files["annotations"])
            if self._download_file("annotations", ann_zip_path):
                self._extract_zip(ann_zip_path, self.root, 'celeba_annotations')
            else:
                self.crop_faces = False
                self.logger.warning("Failed to download annotations, face cropping disabled.")
        elif self.crop_faces:
            self.logger.info("Annotations directory already present, skipping annotation download.")
        
        self.logger.info("CelebA dataset download and extraction completed.")

    def _download_file(self, file_key, save_path):
        """Download a file using gdown.
        
        Parameters
        ----------
        file_key : str
            Key in the urls dictionary ('train' or 'annotations')
        save_path : str
            Path where the file should be saved
            
        Returns
        -------
        bool
            True if download was successful, False otherwise
        """
        # Check if file already exists and is valid
        if os.path.exists(save_path):
            if file_key == "train" or file_key == "annotations":
                try:
                    with zipfile.ZipFile(save_path, 'r') as zf:
                        zf.testzip()  # Test if zip file is valid
                    self.logger.info(f"Valid {file_key} file already exists, skipping download.")
                    return True
                except (zipfile.BadZipFile, zipfile.LargeZipFile):
                    self.logger.warning(f"Existing {file_key} file is corrupted, will re-download.")
                    os.remove(save_path)
        
        # Download with gdown
        url = self.urls[file_key]
        self.logger.info(f"Downloading {file_key} from Google Drive...")
        
        try:
            success = gdown.download(url=url, output=save_path, quiet=False, fuzzy=True)
            if not success:
                raise RuntimeError(f"gdown.download returned False for {file_key}")
                
            # Validate downloaded file
            if file_key == "train" or file_key == "annotations":
                with zipfile.ZipFile(save_path, 'r') as zf:
                    zf.testzip()
                    
            self.logger.info(f"{file_key.capitalize()} download successful and validated.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {file_key}: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return False

    def _extract_zip(self, zip_path, extract_root, target_dir_name):
        """Extract zip file to target directory.
        
        Parameters
        ----------
        zip_path : str
            Path to the zip file
        extract_root : str
            Root directory for extraction
        target_dir_name : str
            Name of the target directory within extract_root
        """
        target_dir = os.path.join(extract_root, target_dir_name)
        
        # Skip extraction if target directory already exists and has content
        if os.path.exists(target_dir) and os.listdir(target_dir):
            self.logger.info(f"Target directory {target_dir} already exists with content, skipping extraction.")
            return
            
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                self.logger.info(f"Extracting {os.path.basename(zip_path)} to {target_dir_name}/...")
                
                # Extract all files and flatten the directory structure
                for member in zf.namelist():
                    # Skip directories
                    if member.endswith('/'):
                        continue
                        
                    # Get the filename (last part of the path)
                    filename = os.path.basename(member)
                    if filename:  # Make sure we have a filename
                        # Extract file directly to target directory
                        source = zf.open(member)
                        target_path = os.path.join(target_dir, filename)
                        
                        with open(target_path, 'wb') as target_file:
                            target_file.write(source.read())
                        source.close()
                
        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid zip file {zip_path}: {e}")
            raise


    def _load_face_annotations(self):
        """Load and parse face landmark annotations."""
        # Look for landmarks file in the annotations directory
        annotations_dir = self.annotations_dir
        landmarks_path = None
        
        # Search for landmarks file in annotations directory
        if os.path.exists(annotations_dir):
            for file in os.listdir(annotations_dir):
                if 'landmark' in file.lower() or 'landmarks' in file.lower():
                    landmarks_path = os.path.join(annotations_dir, file)
                    break
        
        # Fallback to old location if not found in annotations directory
        if not landmarks_path:
            landmarks_path = os.path.join(self.root, 'list_landmarks_align_celeba.txt')
        
        if not os.path.exists(landmarks_path):
            self.logger.warning("Face landmarks file not found. Face cropping will be disabled.")
            self.crop_faces = False
            return
            
        self.logger.info(f"Loading face landmarks from {landmarks_path}...")
        
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
        y_max = int(y_max + margin_y * 0.8)  # Less margin on bottom
        
        return (x_min, y_min, x_max, y_max)

    def _preprocess_images(self, source_dir, dest_dir):
        """Preprocess images to crop faces and resize to target size.
        
        Parameters
        ----------
        source_dir : str
            Directory containing the extracted raw images
        dest_dir : str  
            Directory where processed images will be saved
        """
        import glob
        
        img_paths = glob.glob(os.path.join(source_dir, '*.jpg'))
        
        # Create destination directory
        os.makedirs(dest_dir, exist_ok=True)
        
        target_size = type(self).img_size[1:]  # (H, W)
        
        # Get resize algorithm
        resize_method = getattr(Image, self.resize_algorithm)
        
        self.logger.info(f"Processing {len(img_paths)} images for face cropping...")
        
        for i, img_path in enumerate(img_paths):
            if i % 10000 == 0:
                self.logger.info(f"Processed {i}/{len(img_paths)} images...")
                
            # Load image
            img = Image.open(img_path)
            original_size = img.size  # (width, height)
            
            # Apply face cropping
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
            
            # Save to destination directory
            dest_path = os.path.join(dest_dir, os.path.basename(img_path))
            img_resized.save(dest_path)
        
        self.logger.info(f"Completed processing {len(img_paths)} images.")

    def _load_images_into_memory(self):
        """Eagerly load images into RAM to speed up training."""
        total = len(self.img_paths)
        self.logger.info(f"Loading {total} images into memory...")

        for idx, img_path in enumerate(self.img_paths):
            try:
                img_array = skimage.io.imread(img_path)
                self._img_cache[idx] = img_array
            except Exception as exc:
                self._img_cache[idx] = None
                self.logger.warning(f"Failed to cache image {img_path}: {exc}")

            if (idx + 1) % 5000 == 0:
                self.logger.info(f"Cached {idx + 1}/{total} images...")

        self.logger.info("Finished loading images into memory.")
    
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
        if self._img_cache is not None:
            img = self._img_cache[idx]
        else:
            img = skimage.io.imread(img_path)

        img = self.transforms(img)
        
        # Return image with placeholder label (0) for compatibility
        return img, 0