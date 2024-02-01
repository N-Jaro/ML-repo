import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_patch_positions(filename):
    """Read patch positions from a file."""
    with open(filename, 'r') as f:
        patch_list = [tuple(map(int, line.strip().split(','))) for line in f]
    return patch_list

def plot_patches(image, train_patches, val_patches, test_patches):
    """Plot patches on the image."""
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot training patches in red
    for xmin, ymin, xmax, ymax in train_patches:
        rect = patches.Rectangle((ymin, xmin), ymax - ymin, xmax - xmin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Plot validation patches in green
    for xmin, ymin, xmax, ymax in val_patches:
        rect = patches.Rectangle((ymin, xmin), ymax - ymin, xmax - xmin, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Plot testing patches in blue
    for xmin, ymin, xmax, ymax in test_patches:
        rect = patches.Rectangle((ymin, xmin), ymax - ymin, xmax - xmin, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    plt.show()

# Replace these paths with the actual file paths
train_file = 'train_patches.txt'
val_file = 'val_patches.txt'
test_file = 'test_patches.txt'

# Read patch positions
train_patches = read_patch_positions(train_file)
val_patches = read_patch_positions(val_file)
test_patches = read_patch_positions(test_file)

# Load your image here
image = np.random.rand(1024, 1024, 3)  # Replace with actual image loading

# Plot patches
plot_patches(image, train_patches, val_patches, test_patches)