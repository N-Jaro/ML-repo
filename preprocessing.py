import numpy as np
import random
from rtree import index

def sample_patch(image, patch_size=(256, 256)):
    """Samples a single patch from the image."""
    max_x, max_y = image.shape[0] - patch_size[0], image.shape[1] - patch_size[1]
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return (x, y, x + patch_size[0], y + patch_size[1])  # Return as bounding box

def generate_samples_with_rtree(image, num_train, num_val, patch_size=(256, 256)):
    """Generate training and validation samples using R-tree for overlap checking."""
    idx = index.Index()
    train_patches, val_patches = [], []
    image_top_half = image[:image.shape[0]//2, :]

    # Generate training samples
    for _ in range(num_train):
        while True:
            patch = sample_patch(image_top_half, patch_size)
            if list(idx.intersection(patch)) == []:
                idx.insert(len(train_patches), patch)
                train_patches.append(patch)
                break

    # Generate validation samples
    for _ in range(num_val):
        while True:
            patch = sample_patch(image_top_half, patch_size)
            if list(idx.intersection(patch)) == []:
                val_patches.append(patch)
                break

    if len(train_patches) < num_train or len(val_patches) < num_val:
        print("Could not generate the requested number of samples.")
        return [], []

    return train_patches, val_patches

def moving_window_test_samples(image, overlap, patch_size=(256, 256)):
    """Generate testing samples using moving window."""
    test_patches = []
    step_size = patch_size[0] - overlap

    for x in range(0, image.shape[0] - patch_size[0], step_size):
        for y in range(0, image.shape[1] - patch_size[1], step_size):
            test_patches.append((x, y, x + patch_size[0], y + patch_size[1]))

    return test_patches

def save_patch_positions(patches, filename):
    """Save patch positions to a file."""
    with open(filename, 'w') as f:
        for patch in patches:
            f.write(f"{patch[0]}, {patch[1]}, {patch[2]}, {patch[3]}\n")

# Example Usage
image = np.random.rand(1024, 1024, 3)  # Replace this with actual image loading
train_patches, val_patches = generate_samples_with_rtree(image, 10, 5)
test_patches = moving_window_test_samples(image, 50)

save_patch_positions(train_patches, "train_patches.txt")
save_patch_positions(val_patches, "val_patches.txt")
save_patch_positions(test_patches, "test_patches.txt")