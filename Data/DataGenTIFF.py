
import os
import rasterio
import numpy as np
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
        self.raster_data, self.reference_data, self.image_width, self.image_height = self._load_data()

        self.raster_files = sorted([f for f in os.listdir(data_path) if f.endswith('.tif')])
        if len(self.raster_files) != 8:
            raise ValueError("Incorrect number of raster images. Expected 8.")

        self._load_image_dimensions()
        self.training_patches, self.validation_patches, self.testing_patches = self._create_datasets()

    def _standardize_raster(self, raster_data):
        """Pads or crops a raster to match the reference shape."""
        ref_shape = (self.image_height, self.image_width) 

        new_data = np.zeros(ref_shape, dtype=raster_data.dtype)
        new_data[:raster_data.shape[0], :raster_data.shape[1]] = raster_data  # Copy existing data

        # Handle potential padding
        if ref_shape[0] > raster_data.shape[0]:
            new_data[raster_data.shape[0]:, :] = 0  # Pad bottom
        if ref_shape[1] > raster_data.shape[1]:
            new_data[:, raster_data.shape[1]:] = 0  # Pad right

        return new_data[:ref_shape[0], :ref_shape[1]]  # Crop if necessary

    def _load_data(self):
        """Loads both input rasters and the reference raster."""
        data = {}
        reference_data = None
        image_width = None
        image_height = None

        # Prioritize loading 'reference.tif'
        for file in self.raster_files:
            if file == 'reference.tif': 
                with rasterio.open(os.path.join(self.data_path, file)) as src:
                    reference_data = src.read()
                    image_width, image_height = src.shape
                break  # Exit the loop after loading reference.tif

        # Load remaining rasters        
        for file in self.raster_files:
            if file != 'reference.tif':  
                with rasterio.open(os.path.join(self.data_path, file)) as src:
                    data[file] = self._standardize_raster(src.read())

        return data, reference_data, image_width, image_height

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
        """Generates random patch locations."""
        patches = []
        while len(patches) < num_patches:
            xmin = np.random.randint(0, width - patch_size + 1)
            if top_only:
                ymax = np.random.randint(0, height // 2 - patch_size + 1)  # Top part
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
        while len(patches) < num_patches:
            xmin = np.random.randint(0, width - patch_size + 1)
            if top_only:
                ymax = np.random.randint(0, height // 2 - patch_size + 1)  # Top part
            else:
                ymax = np.random.randint(0, height - patch_size + 1)
            xmax = xmin + patch_size
            ymin = ymax + patch_size

            if len(list(idx.intersection((xmin, ymin, xmax, ymax)))) == 0:
                patches.append([xmin, ymin, xmax, ymax])
        return patches

    def data_generator(self, patch_locations, batch_size, shuffle=True):
        while True:  
            if shuffle:
                np.random.shuffle(patch_locations)

            for i in range(0, len(patch_locations), batch_size):
                batch_patches = patch_locations[i:i + batch_size]
                X_batch = np.zeros((batch_size, self.patch_size, self.patch_size, len(self.raster_files)))
                y_batch = np.zeros((batch_size, self.patch_size, self.patch_size))  # Assuming single-channel reference

                for j, patch in enumerate(batch_patches):
                    xmin, ymin, xmax, ymax = patch

                    for channel, file in enumerate(self.raster_files):
                        if file != 'reference.tif':
                            X_batch[j, :, :, channel] = self.raster_data[file][ymin:ymax, xmin:xmax]  

                    # Load target data from preloaded reference
                    y_batch[j, :, :] = self.reference_data[ymin:ymax, xmin:xmax]

                yield X_batch, y_batch 

    def get_training_generator(self, batch_size=32, shuffle=True):
        """Returns a data generator for the training set."""
        return self.data_generator(self.training_patches, batch_size, shuffle)

    def get_validation_generator(self, batch_size=32, shuffle=True):
        """Returns a data generator for the validation set."""
        return self.data_generator(self.validation_patches, batch_size, shuffle)

    
    def _generate_testing_patches(self, width, height, patch_size, overlap):
        """Generates testing patch locations in a moving window fashion."""
        patches = []
        for x in range(0, width - patch_size + 1, patch_size - overlap):
            for y in range(self.image_height // 2, height - patch_size + 1, patch_size - overlap):  # Bottom part
                patches.append([x, y, x + patch_size, y + patch_size])
        return patches

    def create_test_generator(self, batch_size, overlap):
        """
        Creates a data generator for the test dataset.

        Args:
            batch_size (int): The number of samples per batch.
            overlap (int): Amount of overlap for testing patches.

        Returns:
            Generator: A generator that yields batches of test data.
        """

        test_patches = self._generate_testing_patches(self.image_width, self.image_height, self.patch_size, overlap)

        def test_data_generator():
            while True:
                for i in range(0, len(test_patches), batch_size):
                    batch_patches = test_patches[i:i + batch_size]
                    X_batch = np.zeros((batch_size, self.patch_size, self.patch_size, len(self.raster_files)))

                    for j, patch in enumerate(batch_patches):
                        xmin, ymin, xmax, ymax = patch

                        with rasterio.open(os.path.join(self.data_path, self.raster_files[0])) as src:
                            data = src.read(window=((ymin, ymax), (xmin, xmax)))

                        for channel, file in enumerate(self.raster_files):
                            with rasterio.open(os.path.join(self.data_path, file)) as src:
                                X_batch[j, :, :, channel] = src.read(1, window=((ymin, ymax), (xmin, xmax)))

                    yield X_batch

        return test_data_generator()  # Return the generator instance
