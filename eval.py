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


np.random.seed(4444)

data_generator = DataGenTIFF('/projects/bbym/nathanj/ML-repo/Alexander/', patch_size = 256, num_train_patches = 200, 
                                num_val_patches = 50, overlap = 30)

model = load_model('/projects/bbym/nathanj/ML-repo/rowancreek-Unet-attentionUnet-tf.h5',
                    custom_objects={'multiplication': multiplication,
                                'multiplication2': multiplication2,
                                'dice_coef_loss':dice_coef_loss,
                                'dice_coef':dice_coef,})

                                
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
with rasterio.open(os.path.join(output_filename), 'w', **tiff_profile) as dst:
    dst.write(predict_reconstruct, 1)  # Write to the first band