import os
import wandb 
import shutil
import rasterio
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.data import AUTOTUNE 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.python.ops.numpy_ops import np_config 
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard)
from models.attentionUnet import (dice_coef_loss, dice_coef, jacard_coef, UNET_224, Residual_CNN_block,  multiplication, attention_up_and_concatenate, multiplication2, attention_up_and_concatenate2)
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbCallback
import matplotlib.pyplot as plt
from Data.DataGen import DataGenTIFF

np_config.enable_numpy_behavior()  # For potential tensor operation efficiency
parser = argparse.ArgumentParser(description="Script for training a segmentation model with optional Weights & Biases integration.")

# Model Args
parser.add_argument('--name_id', type=str, default='',
                    help='Identifier for the experiment run (affects save locations).')
parser.add_argument('--gpus', type=int, default=2,
                    help='Number of GPUs to utilize for training.')
parser.add_argument('--project_name', type=str, default='MAML-project',
                    help='Name of the Weights & Biases project.')
parser.add_argument('--model', type=str, default='attentionUnet',
                    help='Segmentation model architecture (e.g., attentionUnet, MultiResUnet)')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training.')
parser.add_argument('--learning_rate', type=float, default=0.00035,
                    help='Initial learning rate for the optimizer.')
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='Optimizer to use (e.g., Adam, SGD, RMSprop)')

#Data Args
parser.add_argument('--data_dir', type=str, default='/projects/bbym/nathanj/ML-repo/Alexander/',
                    help='Directory containing all TIFF files.')
parser.add_argument('--patch_size', type=int, default=256,
                    help='Patch width and height')
parser.add_argument('--num_train_patches', type=int, default=200,
                    help='Number of training patches.')
parser.add_argument('--num_val_patches', type=int, default=50,
                    help='Number of validation patches.')
parser.add_argument('--test_overlap', type=int, default=30,
                    help='Test patch overlap size.')

parser.add_argument('--seed', type=int, default=4444,
                    help='Random seed for reproducibility.')

# parser.add_argument('--augment', type=bool, default=True,
#                     help='Whether to apply data augmentation.')
# parser.add_argument('--rotation_rate', type=float, default=0.4,
#                     help='Probability of applying a random rotation during augmentation.')
# parser.add_argument('--vertical_flip_rate', type=float, default=0.4,
#                     help='Probability of applying a vertical flip during augmentation.')
# parser.add_argument('--horizontal_flip_rate', type=float, default=0.4,
#                     help='Probability of applying a horizontal flip during augmentation.')
# parser.add_argument('--hue_factor', type=float, default=0.2,
#                     help='Maximum amount of hue shift to apply during augmentation.')

args = parser.parse_args()

config = vars(args)

#  Add fixed configurations 
config['prediction_path'] = './predicts_' + config['name_id'] + '/'
config['log_path'] = './logs_' + config['name_id'] + '/'
config['model_path'] = './models_' + config['name_id'] + '/'
config['save_model_path'] = './models_' + config['name_id'] + '/'


print("***Training Initialization ***") 
for k, v in config.items():
    print(k, ":", v)
print("********************************")

np.random.seed(args.seed)

data_generator = DataGenTIFF(args.data_dir, patch_size = args.patch_size, num_train_patches = args.num_train_patches, 
                                num_val_patches = args.num_val_patches, overlap = args.test_overlap)

training_dataset = data_generator.create_train_dataset()
validation_dataset = data_generator.create_validation_dataset()

################################################################
##### Prepare the model configurations #########################
################################################################
#You can change the id for each run so that all models and stats are saved separately.
name_id = args.name_id
prediction_path = config['prediction_path'] 
log_path = config['log_path']
model_path = config['model_path']

# Create the folder if it does not exist
os.makedirs(model_path, exist_ok=True)
os.makedirs(prediction_path, exist_ok=True)

backend = args.model
name = 'Unet-'+ backend
logdir = config['log_path'] + name

if(os.path.isdir(logdir)):
    shutil.rmtree(logdir)

os.makedirs(logdir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

print('model location: '+ model_path + name + '.h5')

# wandb.init( 
#     project=args.project_name,  
#     sync_tensorboard=True,    
#     name=args.name_id,       
#     config=config           
# )

# Initialize the MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def create_unet(input_shape, num_classes):
    """Creates a small U-Net model for image segmentation.

    Args:
        input_shape: The shape of the input images (height, width, channels).
        num_classes: Number of classes to predict (for segmentation output).

    Returns:
        A Keras model representing the U-Net architecture.
    """

    # Encoder (Downsampling)
    inputs = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck 
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    # Decoder (Upsampling)
    up1 = layers.Conv2DTranspose(64, 2, strides=(2,2), padding='same')(conv3)
    concat1 = layers.concatenate([up1, conv2], axis=3)  # Skip connection
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat1)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    up2 = layers.Conv2DTranspose(32, 2, strides=(2,2), padding='same')(conv4)
    concat2 = layers.concatenate([up2, conv1], axis=3) 
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat2)
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid' if num_classes==1 else 'softmax')(conv5)  

    return Model(inputs=inputs, outputs=outputs)

# Example usage 
input_shape = (256, 256, 8)  # Example shape
num_classes = 1  # Adjust for your segmentation task 

# Define a model within the strategy's scope
with strategy.scope():
    # if os.path.isfile(model_path + name + '.h5'):
    #     model = load_model(model_path+name+'.h5',
    #                 custom_objects={'multiplication': multiplication,
    #                             'multiplication2': multiplication2,
    #                             'dice_coef_loss':dice_coef_loss,
    #                             'dice_coef':dice_coef,})
    # else:
    # if (args.model == 'attentionUnet'):
    model = create_unet(input_shape, num_classes)
    model.compile(optimizer = Adam(learning_rate=0.0001,clipnorm=1.0),
                    loss = dice_coef_loss,
                    metrics = [dice_coef,'accuracy'])

# Add L2 regularization to certain layers:  
for layer in model.layers:  
    if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):  
        layer.kernel_regularizer = l2(0.001)  # l2 regularizer with strength of 0.001

# define hyperparameters and callback modules
patience = 3
maxepoch = 500
batch_size = args.batch_size
# callbacks = [
#                 ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=patience, min_lr=1e-9, verbose=1, mode='min'),
#                 EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
#                 ModelCheckpoint(model_path+name+'.h5', monitor='val_loss', save_best_only=True, verbose=0),
#                 TensorBoard(log_dir=logdir),
#                 WandbMetricsLogger(log_freq="epoch"),
#                 WandbCallback(save_model=True, monitor="val_loss") 
#             ]
# train_history = model.fit(training_dataset, validation_data = validation_dataset, batch_size = batch_size, 
#                             epochs = maxepoch, verbose=1, callbacks = callbacks)
train_history = model.fit(training_dataset, validation_data = validation_dataset, batch_size = batch_size, 
                            epochs = maxepoch, verbose=1)

# # Log additional metrics at the end of training
# wandb.log({
#     "final_train_loss": train_history.history['loss'][-1],
#     "final_val_loss": train_history.history['val_loss'][-1],
#     "final_val_accuracy": train_history.history['val_accuracy'][-1]
# })

test_dataset = data_generator.create_test_dataset()

prediction = model.predict(test_dataset)

print(prediction.shape)

predict_reconstruct = data_generator.reconstruct_predictions(prediction[:,:,:,0])

# Prepare metadata for saving as TIFF
tiff_profile = {
    'driver': 'GTiff',
    'dtype': predict_reconstruct.dtype,
    'nodata': None,  # Adjust if you have specific no-data values
    'width': predict_reconstruct.shape[1],
    'height': predict_reconstruct.shape[0],
    'count': 1,  # Assuming a single-band image 
}

# Save to TIFF
output_filename = 'Alexander_predict.tif'
with rasterio.open(os.path.join(prediction_path,output_filename), 'w', **tiff_profile) as dst:
    dst.write(predict_reconstruct, 1)  # Write to the first band

