import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import load_model
import cv2
import pickle
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from doc import train
import json
import datetime
import skimage.draw
from skimage.transform import resize
ROOT_DIR = os.path.abspath("../../")
doc_dir = os.path.abspath("../../datasets/doc/")
input_dir = os.path.join(ROOT_DIR,"./train/")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import keras
import keras.backend as K
import keras.layers as KL
import keras.losses as LS
import keras.engine as KE
import keras.models as KM
from scipy import ndimage
import tensorflow.contrib.eager as tfe
from tensorflow.python.eager.context import eager_mode
# tf.enable_eager_execution()
# from tensorflow.python.client import device_lib
# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')
config = train.Config()
DOC_DIR = os.path.join(ROOT_DIR, "datasets/doc/")



mode = "inference"
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0



def tilted_loss(y,f,q = 0.5):
    gt_y = y[:,1:]
    f = f[:,:-1]
    e = (gt_y - f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    gt_y = y_true[:,1:]
    f = y_pred[:,:-1]
    return K.mean(K.abs(gt_y - f), axis=-1)

def smooth_l2_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    gt_y = y_true[:,1:]
    f = y_pred[:,:-1]
    return K.mean(K.abs(gt_y - f)**2, axis=-1)
def build_rpn_targets(image_shape, gt_boxes):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    columnIndex = 0
    temp_gt_boxes = gt_boxes[gt_boxes[:,columnIndex].argsort()]
    rpn_bbox = np.zeros(gt_boxes.shape)
    rpn_out = np.zeros((gt_boxes.shape[0]+1,gt_boxes.shape[1]))
    for i in range(gt_boxes.shape[0]):
        rpn_bbox[i][0] = temp_gt_boxes[i][0] / image_shape[0]
        rpn_bbox[i][1] = temp_gt_boxes[i][1] / image_shape[1]
        rpn_bbox[i][2] = (temp_gt_boxes[i][2] - temp_gt_boxes[i][0]) / image_shape[0]
        rpn_bbox[i][3] = (temp_gt_boxes[i][3] - temp_gt_boxes[i][1]) / image_shape[1]
    for i in range(1,gt_boxes.shape[0]):
        for j in range(4):
            rpn_out[i,j] = rpn_bbox[i][j] - rpn_bbox[i-1][j]
    return rpn_out

input_image = KL.Input(
            shape=[None, None, 512], name="input_feature")

decoder_inputs = KL.Input(
        shape=[1,4], name="input_rpn_bbox", dtype=tf.float32)
# gap
encoder_inputs = KL.GlobalAveragePooling2D(data_format=None)(input_image)
# calculate rpn_bbox here


encoder1 = KL.Dense(4, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',name = "encode1")
# encoder2 = KL.Dense(4, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',name = "encode2")

decoder = KL.GRU(4,activation = 'sigmoid', return_sequences=True, return_state=True, input_shape = (None,4),name = "decode")

state_h = encoder1(encoder_inputs)
# state_c = encoder2(encoder_inputs)
# encoded = [state_h, state_c]

encoder_model = KM.Model(input_image, state_h)
decoder_states_input_h = KL.Input(shape=[4], name="input_state_h", dtype=tf.float32)
# decoder_states_input_c = KL.Input(shape=[4], name="input_state_c", dtype=tf.float32)
# decoder_states_inputs = [decoder_states_input_h, decoder_states_input_c]
decoder_outputs, final_state_h = decoder(
    decoder_inputs, initial_state=decoder_states_input_h)
decoder_states = final_state_h
decoder_model = KM.Model(
    [decoder_inputs,decoder_states_input_h],
    [decoder_outputs,decoder_states])
weights_path = "../../mask_rcnn_object_0080.h5"
print("Loading weights ", weights_path)

def load_weights(model,filepath, by_name=False, exclude=None):
    """Modified version of the corresponding Keras function with
    the addition of multi-GPU support and the ability to exclude
    some layers from loading.
    exclude: list of layer names to exclude
    """
    import h5py
    # Conditional import to support versions of Keras before 2.2
    # TODO: remove in about 6 months (end of 2018)
    try:
        from keras.engine import saving
    except ImportError:
        # Keras before 2.2 used the 'topology' namespace.
        from keras.engine import topology as saving

    if exclude:
        by_name = True

    if h5py is None:
        raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    keras_model = model
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
        else keras_model.layers

    # Exclude some layers
    if exclude:
        layers = filter(lambda l: l.name not in exclude, layers)

    if by_name:
        saving.load_weights_from_hdf5_group_by_name(f, layers)
    else:
        saving.load_weights_from_hdf5_group(f, layers)
    if hasattr(f, 'close'):
        f.close()

    # Update the log directory
    
load_weights(encoder_model,weights_path,by_name=True)
load_weights(decoder_model,weights_path,by_name=True)


# Create model in inference mode
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    
    # input_seq = tf.convert_to_tensor(input_seq, dtype=tf.float32)

    # input_seq = KL.Lambda(
        # lambda t: tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3]]))(input_seq)
    print(input_seq.shape)
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 4))
    target_seq[0,0,:] = np.array([0.1,0.1,0.3,0.4])
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h = decoder_model.predict(
            [target_seq,states_value])

        # Sample a token
        decoded_sentence.append(output_tokens)
        print(output_tokens, h)
        # Exit condition: either hit max length
        # or find stop character.
        if (np.all(output_tokens[0,0,:] == np.array([0,0,0,0]))) or (len(decoded_sentence) >=2 and (np.sum(output_tokens[0,0,:] - decoded_sentence[-2]) <=0.000001)):
        # if (np.all(output_tokens[0,0,:] == np.array([0,0,0,0]))) or (len(decoded_sentence) >=2 and output_tokens[0,0,0] <=0.01):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = output_tokens

        # Update states
        states_value = h

    return decoded_sentence
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "square"
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.3
    PRE_NMS_LIMIT = 4000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_INFERENCE = 1000
    

config = InferenceConfig()
# config.display()

# GPU for training.

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

# with sess.as_default():
# Load validation dataset
dataset = train.Dataset()
dataset.load_data(DOC_DIR, "test")
# Must call before using the dataset
dataset.prepare()
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
# for i, info in enumerate(dataset.class_info):
    # print("{:3}. {:50}".format(i, info['name']))

cnt=0
# Must call before using the dataset
all_images_test=dataset.image_ids
instance_count=[0]*10
ind = 0
# for ind in range(len(all_images_test)):
cnt+=1
image_id=all_images_test[ind]
print(ind," : ",image_id)
image, image1, image_meta, class_ids, bbox, mask =    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
batch_images1 = np.zeros((1,) + image1.shape, dtype=np.float32)
batch_images1[0] = image1.astype(np.float32)

info = dataset.image_info[image_id]
img_name=info['id']
print(img_name)
rpn_bbox = build_rpn_targets(image.shape,bbox)
results = decode_sequence(batch_images1)
results = np.array(results)
print((results.shape)
    # print(results)