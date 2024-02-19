import tensorflow as tf
from keras import backend as K

# Use dice coefficient function as the loss function 
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

# Jacard coefficient
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def total_variation_loss(y_pred):
    dx = tf.reduce_sum(tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]))
    dy = tf.reduce_sum(tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))
    return dx + dy

def smooth_dice_combined_loss(y_true, y_pred):
    weight_dice = 0.7
    weight_tv = 0.3
    return weight_dice * (1 - dice_coef(y_true, y_pred)) + weight_tv * total_variation_loss(y_pred)

# calculate loss value
def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

# calculate loss value
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
