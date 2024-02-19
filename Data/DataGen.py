import os
import rasterio
import numpy as np
import tensorflow as tf
from rtree import index

class DataGenTIFF:
    def __init__(self, data_path, patch_size=256, num_train_patches=200, num_val_patches=100, overlap=30):
        """
        Initializes the DataGenTIFF class.

        Args:
            data_path (str): Path to the directory containing raster images.
            patch_size (int): Size of the patches (assumes square patches).
            num_train_patches (int): Number of training patches to generate.
            num_val_patches (int): Number of validation patches to generate.
            overlap (int): Amount of overlap for testing patches.
        """
        self.data_path = data_path
        self.patch_size = patch_size
        self.num_train_patches = num_train_patches
        self.num_val_patches = num_val_patches
        self.overlap = overlap
        

        # Collect all TIFF files, explicitly excluding 'reference.tif'
        self.raster_files = sorted([f for f in os.listdir(data_path) 
                                    if f.endswith('.tif') and f != 'reference.tif'])
        if len(self.raster_files) != 8:  # Updated expected count
            raise ValueError("Incorrect number of raster images. Expected 8 (excluding reference.tif)")
        
        self.raster_data, self.reference_data = self._load_data()
        self.training_patches, self.validation_patches = self._create_datasets()

    def _standardize_raster(self, raster_data):
        """Pads or crops a raster to match the reference shape."""
        ref_shape = (self.image_height, self.image_width) 

        new_data = np.zeros(ref_shape, dtype=raster_data.dtype)
        new_data[:raster_data.shape[0], :raster_data.shape[1]] = raster_data  

        # Handle potential padding
        if ref_shape[0] > raster_data.shape[0]:
            new_data[raster_data.shape[0]:, :] = 0  
        if ref_shape[1] > raster_data.shape[1]:
            new_data[:, raster_data.shape[1]:] = 0 

        return new_data[:ref_shape[0], :ref_shape[1]] 

    def _load_data(self):
        """Loads both input rasters and the reference raster."""
        data = {}
        reference_data = None
        image_width = None
        image_height = None

        for file in self.raster_files:
            if file == 'reference.tif': 
                with rasterio.open(os.path.join(self.data_path, file)) as src:
                    reference_data = src.read()
                    print("Reference.tf")
                    self.image_width, self.image_height = src.shape
                break  

        for file in self.raster_files:
            if file != 'reference.tif':  
                with rasterio.open(os.path.join(self.data_path, file)) as src:
                    data[file] = self._standardize_raster(src.read())

        return data, reference_data

    def _create_datasets(self):
        """Creates training and validation datasets from raster images."""
        # Generate training patches
        training_patches = self._generate_random_patches(
            self.image_width, self.image_height, self.patch_size, self.num_train_patches, top_only=True
        )
        # Generate validation patches (non-overlapping with training)
        validation_patches = self._generate_nonoverlapping_patches(
            self.image_width, self.image_height, self.patch_size, self.num_val_patches, training_patches, top_only=True
        )
        if len(validation_patches) < self.num_val_patches:
            print(f"Warning: Generated only {len(validation_patches)} validation patches.")
        return training_patches, validation_patches

    def _generate_random_patches(self, width, height, patch_size, num_patches, top_only=False):
        patches = []
        while len(patches) < num_patches:
            xmin = np.random.randint(0, width - patch_size + 1)
            if top_only:
                ymax = np.random.randint(0, height // 2 - patch_size + 1)  
            else:
                ymax = np.random.randint(0, height - patch_size + 1)
            xmax = xmin + patch_size
            ymin = ymax + patch_size
            patches.append([xmin, ymin, xmax, ymax])
        return patches
    
    def _generate_nonoverlapping_patches(self, width, height, patch_size, num_patches, existing_patches, top_only=False):
        """Generates non-overlapping patch locations."""
        idx = index.Index()
        for i, patch in enumerate(existing_patches):
            idx.insert(i, patch)

        patches = []
        max_attempts = 100000  # Maximum attempts to find non-overlapping patches

        for _ in range(max_attempts):
            if len(patches) >= num_patches:
                break 

            xmin = np.random.randint(0, width - patch_size + 1)
            if top_only:
                ymax = np.random.randint(0, height // 2 - patch_size + 1) 
            else:
                ymax = np.random.randint(0, height - patch_size + 1)
            xmax = xmin + patch_size
            ymin = ymax + patch_size

            if len(list(idx.intersection((xmin, ymin, xmax, ymax)))) == 0:
                patches.append([xmin, ymin, xmax, ymax])

        return patches 

    def tf_data_generator(self, patch_locations, batch_size, shuffle=True):
        """Data generator using TensorFlow's Dataset API."""
        def load_and_process_patch(patch):
            xmin, ymin, xmax, ymax = patch
            x_data = tf.stack([self.raster_data[file][ymin:ymax, xmin:xmax] 
                                for file in self.raster_files if file != 'reference.tif'], axis=-1)
            y_data = self.reference_data[ymin:ymax, xmin:xmax] 
            return x_data, y_data

        dataset = tf.data.Dataset.from_tensor_slices(patch_locations)
        if shuffle:
            dataset = dataset.shuffle(len(patch_locations)) 
        dataset = dataset.map(load_and_process_patch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)   
        return dataset 

    def get_training_generator(self, batch_size=32, shuffle=True):
        return self.tf_data_generator(self.training_patches, batch_size, shuffle)

    def get_validation_generator(self, batch_size=32, shuffle=True):
        return self.tf_data_generator(self.validation_patches, batch_size, shuffle)

    def _generate_testing_patches(self): 
        """Generates testing patch locations (bottom part only)."""
        patches = []
        for x in range(0, self.image_width - self.patch_size + 1, self.patch_size - self.overlap):
            for y in range(self.image_height // 2, self.image_height - self.patch_size + 1, self.patch_size - self.overlap):  
                patches.append([x, y, x + self.patch_size, y + self.patch_size])
        return patches

    def create_test_generator(self, batch_size):
        def test_data_generator():

            def load_patches(xmin, ymin, xmax, ymax):
                x_data = tf.stack([self.raster_data[file][ymin:ymax, xmin:xmax] 
                                for file in self.raster_files if file != 'reference.tif'], axis=-1)
                return x_data

            for patch in self._generate_testing_patches():
                xmin, ymin, xmax, ymax = patch
                X_batch = tf.data.Dataset.from_tensors(0).repeat(batch_size) 
                X_batch = X_batch.map(lambda _: load_patches(xmin, ymin, xmax, ymax)) 
                yield tf.squeeze(X_batch)  

        return test_data_generator() 