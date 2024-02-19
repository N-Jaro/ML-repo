import os
import rasterio
import numpy as np
from PIL import Image
import tensorflow as tf
from rtree import index
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class DataGenTIFF:
    def __init__(self, data_path,batch_size=32, patch_size=256, num_train_patches=200, num_val_patches=100, overlap=30):
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
        self.batch_size = batch_size
        self.num_train_patches = num_train_patches
        self.num_val_patches = num_val_patches
        self.overlap = overlap
        self.image_width = 0
        self.image_height = 0

        # Print out the configuration
        print("    GenTIFF Initialization:")
        print(f"Data Path: {self.data_path}")
        print(f"Patch Size: {self.patch_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Training Patches: {self.num_train_patches}")
        print(f"Validation Patches: {self.num_val_patches}")
        print(f"Overlap: {self.overlap}")

        # Collect all TIFF files, explicitly excluding 'reference.tif'
        self.raster_files = sorted([f for f in os.listdir(data_path) if f.endswith('.tif')])
        if len(self.raster_files) != 9:  # Updated expected count
            raise ValueError("Incorrect number of raster images. Expected 9 ( 8 + reference.tif)")
        
        self.raster_data, self.reference_data = self._load_data()
        self.training_patches, self.validation_patches = self._create_datasets()

    def _standardize_raster(self, raster_data):
        """Pads or crops a raster to match the reference shape."""
        # print(self.image_height, self.image_width)
        ref_shape = (self.image_height, self.image_width) 

        new_data = np.zeros(ref_shape, dtype=raster_data.dtype)
        new_data = raster_data[:ref_shape[0], :ref_shape[1]] 

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

        for file in self.raster_files:
            if file == 'reference.tif': 
                reference_data = np.array(Image.open(os.path.join(self.data_path, file)))
                self.image_height, self.image_width = reference_data.shape
                break  

        for file in self.raster_files:
            if file != 'reference.tif':  
                raster_array = np.array(Image.open(os.path.join(self.data_path, file))).astype(float)

                # Modify pixels less than -500 (using a single operation)
                raster_array[raster_array < -1000] = np.nan

                # Quantile normalization
                q1 = np.nanpercentile(raster_array, 5)  # Lower quantile (e.g., 5th percentile)
                q3 = np.nanpercentile(raster_array, 95)  # Upper quantile (e.g., 95th percentile)
                iqr = q3 - q1
                raster_array = np.clip(raster_array, q1, q3)  # Clip outliers
                raster_array = (raster_array - q1) / iqr  # Normalize

                data[file] = self._standardize_raster(raster_array) 

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
    
    def is_patch_valid(self, ymin, ymax, xmin, xmax):
        """Checks if all pixel values within a patch are non-NaN."""

        file = self.raster_files[0]
        patch_data = self.raster_data[file][ymin:ymax, xmin:xmax]

        # Correct NaN check
        if np.isnan(patch_data).any(): 
            return False  # Patch invalid if any pixel is NaN
        return True  # Patch valid if no NaNs are found
    
    def _generate_random_patches(self, width, height, patch_size, num_patches, top_only=False):
        patches = []
        while len(patches) < num_patches:
            xmin = np.random.randint(0, width - patch_size + 1)
            if top_only:
                ymin = np.random.randint(0, height // 2 - patch_size + 1)  
            else:
                ymin = np.random.randint(0, height - patch_size + 1)
            xmax = xmin + patch_size
            ymax = ymin + patch_size
            
            if self.is_patch_valid(ymin, ymax, xmin, xmax):
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
                ymin = np.random.randint(0, height // 2 - patch_size + 1) 
            else:
                ymin = np.random.randint(0, height - patch_size + 1)
            xmax = xmin + patch_size
            ymax = ymin + patch_size

                    # Validity Check
            if self.is_patch_valid(ymin, ymax, xmin, xmax) and len(list(idx.intersection((xmin, ymin, xmax, ymax)))) == 0:
                patches.append([xmin, ymin, xmax, ymax])

        return patches 

    def _load_and_process_patch(self, xmin,ymin,xmax,ymax):
        x_data = tf.stack([self.raster_data[file][ymin:ymax, xmin:xmax] 
                        for file in self.raster_files if file != 'reference.tif'], axis=-1)
        y_data = self.reference_data[ymin:ymax, xmin:xmax] 
        return x_data, y_data

    def create_train_dataset(self):
        """Creates a TensorFlow Dataset object for the training set."""
        def generator():
            patches = self.training_patches
            for xmin,ymin,xmax,ymax in patches:
                data, label = self._load_and_process_patch(xmin,ymin,xmax,ymax)
                yield data, label

        dataset = tf.data.Dataset.from_generator(
            generator,  # Call the generator
            output_signature=(
                tf.TensorSpec(shape=(256, 256, len(self.raster_files) - 1), dtype=tf.float32), 
                tf.TensorSpec(shape=(256, 256), dtype=tf.float32)
            )
        )

        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def create_validation_dataset(self):
        """Creates a TensorFlow Dataset object for the training set."""
        def generator():
            patches = self.validation_patches
            for xmin,ymin,xmax,ymax in patches:
                data, label = self._load_and_process_patch(xmin,ymin,xmax,ymax)
                yield data, label

        dataset = tf.data.Dataset.from_generator(
            generator,  # Call the generator
            output_signature=(
                tf.TensorSpec(shape=(256, 256, len(self.raster_files) - 1), dtype=tf.float32), 
                tf.TensorSpec(shape=(256, 256), dtype=tf.float32)
            )
        )

        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def _generate_testing_patches(self): 
        """Generates testing patch locations (bottom part only)."""
        patches = []
        for x in range(0, self.image_width - self.patch_size + 1, self.patch_size - self.overlap):
            for y in range(self.image_height // 2, self.image_height - self.patch_size + 1, self.patch_size - self.overlap):  
                patches.append([x, y, x + self.patch_size, y + self.patch_size])
        return patches

    def create_validation_dataset(self):
        """Creates a TensorFlow Dataset object for the training set."""
        def generator():
            patches = self._generate_testing_patches()
            for xmin,ymin,xmax,ymax in patches:
                data, label = self._load_and_process_patch(xmin,ymin,xmax,ymax)
                yield data, label

        dataset = tf.data.Dataset.from_generator(
            generator,  # Call the generator
            output_signature=(
                tf.TensorSpec(shape=(256, 256, len(self.raster_files) - 1), dtype=tf.float32), 
                tf.TensorSpec(shape=(256, 256), dtype=tf.float32)
            )
        )

        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def reconstruct_predictions(self, predictions):
        """Reconstructs a full image from predicted patches obtained from the test data.

        Args:
            predictions: A NumPy array of shape (n, 256, 256) where n is the number of 
                        predicted patches.

        Returns:
            A NumPy array representing the reconstructed image.
        """

        reconstructed_image = np.zeros((self.image_height, self.image_width), dtype=predictions.dtype)
        patch_count = np.zeros_like(reconstructed_image)  # Keep track of patch overlaps

        patch_num = 0
        for x in range(0, self.image_width - self.patch_size + 1, self.patch_size - self.overlap):
            for y in range(self.image_height // 2, self.image_height - self.patch_size + 1, self.patch_size - self.overlap): 
                reconstructed_image[y : y + self.patch_size, x : x + self.patch_size] += predictions[patch_num, :, :]
                patch_count[y : y + self.patch_size, x : x + self.patch_size] += 1
                patch_num += 1

        # Average out overlapped regions
        reconstructed_image /= patch_count

        return reconstructed_image
    
    def visualize_patches(self):
        """Draws squares for training and validation patches on an image-sized canvas."""

        fig, ax = plt.subplots(figsize=(self.image_width / 100, self.image_height / 100))  #  Scale according to preference
        # Display the image as the background
        ax.imshow(self.reference_data, cmap='gray', vmin=0, vmax=1)

        # Training patches in green
        for patch in self.training_patches:
            xmin, ymin, xmax, ymax = patch
            rect = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        # Validation patches in red
        for patch in self.validation_patches:
            xmin, ymin, xmax, ymax = patch
            rect = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.set_xlim(0, self.image_width)
        ax.set_ylim(self.image_height, 0)  # Invert y-axis for image-like coordinates 
        ax.set_aspect('equal')  

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Patch Locations')
        plt.show()

    def visualize_map(self, indx):
        """Draws squares for training and validation patches on an image-sized canvas."""

        fig, ax = plt.subplots(figsize=(self.image_width / 100, self.image_height / 100))  #  Scale according to preference
        file = self.raster_files[indx]
        plt.imshow(self.raster_data[file], cmap='gray',vmin=0, vmax=1)
        # plt.title('Raster at ', indx)
        plt.show()