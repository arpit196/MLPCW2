#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar


# In[3]:


#!mkdir train


# In[ ]:


#!tar -xvf 'VOCtrainval_11-May-2012.tar' -C 'train'


# In[1]:


import sklearn as sk


# In[2]:

import preprocessing
import tensorflow as tf
import os
import cv2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow.keras.backend as K
import random
from random import choice


# In[3]:


from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D,     Dropout, Conv2DTranspose, Cropping2D, Add, UpSampling2D, LayerNormalization, BatchNormalization


# In[4]:


from os import path
from matplotlib import pyplot as plt


# In[5]:


import keras


# In[6]:


from tensorflow.keras import layers
from tensorflow.keras import activations


# In[7]:


import keras as k
import PIL
from PIL import Image
from tensorflow.keras.utils import to_categorical


# In[8]:


import shutil


# In[9]:


from keras.preprocessing.image import ImageDataGenerator
tf.config.experimental_run_functions_eagerly(True)


# In[10]:


xtrain = []
ytrain_bb = []
labels = []
label_list = []


# In[11]:


def label_enc(label):
  global label_list
  if label not in label_list:
    label_list.append(label)
    return len(label_list)
  else:
    return label_list.index(label)


# In[12]:


def gaussian(sample, kernel_size):
    #sample = np.array(sample)
    # blur the image with a 50% chance
    prob = np.random.random_sample()
    sigma = (2 - 1) * np.random.random_sample() + 1
    sample = cv2.GaussianBlur(sample, (kernel_size, kernel_size), sigma)
    return sample


# In[13]:


def gaussian_blur(v1):
    k_size = int(v1.shape[1] * 0.1)  # kernel size is set to be 10% of the image height/width
    if(k_size%2==0):
        k_size = k_size + 1
    v1 = gaussian(v1, kernel_size=k_size)
    #[v1, ] = tf.py_function(gaussian_ope, [v1], [tf.float32])
    #[v2, ] = tf.py_function(gaussian_ope, [v2], [tf.float32])
    return v1


# In[16]:


def label_enc(classname):
  global labels
  if classname in labels:
    return labels.index(classname)+1
    #return tf.one_hot(indices = labels.index(classname)+1, depth = len(labels)+1)
  else:
    return 0
    #return tf.one_hot(indices = 0, depth = len(labels)+1)



# In[24]:


def flip_random_left_right(image, anno):
    '''
    :param image: [height, width, channel]
    :return:
    '''
    flag = random.randint(0, 1)
    if flag:
        return cv2.flip(image, 1), cv2.flip(anno, 1)
    return image, anno


def random_pad_crop(image, anno, crop_height, crop_width, ignore_label):

    image = image.astype(np.float32)

    height, width = anno.shape

    #padded_image = np.pad(image, ((0, np.maximum(height, HEIGHT) - height), (0, np.maximum(width, WIDTH) - width), (0, 0)), mode='constant', constant_values=_MEAN_RGB)

    padded_image_r = np.pad(image[:, :, 0], ((0, np.maximum(height, crop_height) - height), (0, np.maximum(width, crop_width) - width)), mode='constant')
    padded_image_g = np.pad(image[:, :, 1], ((0, np.maximum(height, crop_height) - height), (0, np.maximum(width, crop_width) - width)), mode='constant')
    padded_image_b = np.pad(image[:, :, 2], ((0, np.maximum(height, crop_height) - height), (0, np.maximum(width, crop_width) - width)), mode='constant')
    padded_image = np.zeros(shape=[np.maximum(height, crop_height), np.maximum(width, crop_width), 3], dtype=np.float32)
    padded_image[:, :, 0] = padded_image_r
    padded_image[:, :, 1] = padded_image_g
    padded_image[:, :, 2] = padded_image_b

    padded_anno = np.pad(anno, ((0, np.maximum(height, crop_height) - height), (0, np.maximum(width, crop_width) - width)), mode='constant', constant_values=ignore_label)

    y = random.randint(0, np.maximum(height, crop_height) - crop_height)
    x = random.randint(0, np.maximum(width, crop_width) - crop_width)

    cropped_image = padded_image[y:y+crop_height, x:x+crop_width, :]
    cropped_anno = padded_anno[y:y+crop_height, x:x+crop_width]

    return cropped_image, cropped_anno


def random_resize(image, anno, scales):
    height, width = anno.shape[:2]

    scale = choice(scales)
    scale_image = cv2.resize(image, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_LINEAR)
    scale_anno = cv2.resize(anno, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_NEAREST)

    return scale_image, scale_anno


def mean_substraction(image, rgb_mean):
    substraction_mean_image = np.zeros_like(image, dtype=np.float32)
    substraction_mean_image[:, :, 0] = image[:, :, 0] - rgb_mean[0]
    substraction_mean_image[:, :, 1] = image[:, :, 1] - rgb_mean[1]
    substraction_mean_image[:, :, 2] = image[:, :, 2] - rgb_mean[2]

    return substraction_mean_image


def augment(img, anno, crop_height, crop_width, ignore_label, random_scales, scales, random_mirror):

    if random_scales:
        scale_img, scale_anno = random_resize(img, anno, scales)
    else:
        scale_img, scale_anno = img, anno

    scale_img = scale_img.astype(np.float32)

    cropped_image, cropped_anno = random_pad_crop(scale_img, scale_anno, crop_height, crop_width, ignore_label)

    if random_mirror:
        cropped_image, cropped_anno = flip_random_left_right(cropped_image, cropped_anno)

    #substracted_img = mean_substraction(cropped_image, rgb_mean)

    return cropped_image, cropped_anno



# In[20]:


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


# In[21]:


from tensorflow import keras


# In[35]:


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


# In[22]:


import xml.etree.ElementTree as ET

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(float(boxes.find("bndbox/ymin").text))
        xmin = int(float(boxes.find("bndbox/xmin").text))
        ymax = int(float(boxes.find("bndbox/ymax").text))
        xmax = int(float(boxes.find("bndbox/xmax").text))
        label = str(boxes.find("name").text)
        list_with_single_boxes = {"x1": xmin, "y1": ymin, "x2": xmax, "y2": ymax, "label": label}
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_boxes


# In[ ]:

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)

def preprocess(X, y):
  X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.10)
  model1 = model()
  checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
  return X_train, y_train, X_test, y_test, model1


# In[36]:
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255
IMAGE_SIZE = 512

mydict={0:0, 14:1,19:2,112:3,220:0, 147:5, 52:6, 57:7, 19:8, 132:9, 75:10, 33:11, 38:11, 89:12, 150:13, 47:14, 33:15, 52:16,113:17,47:18, 37:19, 72:20, 255:0 }
keys = list(mydict.keys())
values = [mydict[k] for k in keys]
table = tf.lookup.StaticHashTable(
  tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int32, value_dtype=tf.int32),  default_value=0
)

def lab(image, label):
    print(table.lookup(label))
    label2 = tf.cast(tf.image.convert_image_dtype(table.lookup(label), dtype=tf.int32),dtype=tf.int32)
    #tf.py_function(func=disp, inp=[image, label2], Tout=tf.int32)
    return image, label2

def get_filenames(is_training, data_dir):
  """Return a list of filenames.
  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.
  Returns:
    A list of file names.
  """
  if is_training:
    return [os.path.join(data_dir, 'voc_train.record')]
  else:
    return [os.path.join(data_dir, 'voc_val.record')]

def parse_record(raw_record):
  """Parse PASCAL image and label from a tf record."""
  keys_to_features = {
      'image/height':
      tf.io.FixedLenFeature((), tf.int64),
      'image/width':
      tf.io.FixedLenFeature((), tf.int64),
      'image/encoded':
      tf.io.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
      tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
      'label/encoded':
      tf.io.FixedLenFeature((), tf.string, default_value=''),
      'label/format':
      tf.io.FixedLenFeature((), tf.string, default_value='png'),
  }

  parsed = tf.io.parse_single_example(raw_record, keys_to_features)

  # height = tf.cast(parsed['image/height'], tf.int32)
  # width = tf.cast(parsed['image/width'], tf.int32)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]), 3)
  image = tf.cast(tf.image.convert_image_dtype(image, dtype=tf.uint8),dtype=tf.float32)
  image.set_shape([None, None, 3])

  label = tf.image.decode_image(
      tf.reshape(parsed['label/encoded'], shape=[]), 1)
  label = tf.cast(tf.image.convert_image_dtype(label, dtype=tf.uint8),dtype=tf.int32)
  label.set_shape([None, None, 1])

  return image, label

def preprocess_image(image, label, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Randomly scale the image and label.
    image, label = preprocessing.random_rescale_image_and_label(
        image, label, _MIN_SCALE, _MAX_SCALE)

    # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
    image, label = preprocessing.random_crop_or_pad_image_and_label(
        image, label, IMAGE_SIZE, IMAGE_SIZE, _IGNORE_LABEL)

    # Randomly flip the image and label horizontally.
    image, label = preprocessing.random_flip_left_right_image_and_label(
        image, label)

    image.set_shape([None, None, 3])
    label.set_shape([None, None, 1])
    
  image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE],method='nearest')
  label = tf.image.resize(images=label, size=[IMAGE_SIZE, IMAGE_SIZE])

  image = preprocessing.mean_image_subtraction(image)
  #label = tf.where(label > 20, 0, label)
  image = tf.cast(image,dtype=tf.float32)
  label = tf.cast(label,dtype=tf.float32)
  
  label = tf.where(label > 20.0, 0.0, label)
  image = tf.cast(image,dtype=tf.float32)
  print(tf.shape(image))
  print(tf.unique(tf.reshape(label,[-1])))

  return image, label


def input_fn(is_training, data_dir, batch_size, model, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.
  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
  Returns:
    A tuple of images and labels.
  """
  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, '/home/arpit_manu6/dataset'))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=10582)

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
          lambda image, label: lab(image, label))
  dataset = dataset.map(
          lambda image, label: preprocess_image(image, label, is_training))
    #dataset = dataset.prefetch(batch_size)
    #print(tf.shape(dataset))
    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
  #dataset = dataset.repeat(num_epochs)
    #dataset = dataset.unbatch()
  dataset = dataset.batch(batch_size,drop_remainder=True)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    #iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    #images, labels = iterator.get_next()
    #labels = tf.where(labels > 20, 0, labels)
    #print(tf.shape(images))
    #print(tf.shape(labels))
    #print(tf.unique(tf.reshape(labels,-1)))
  return dataset


def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


# In[28]:


import segmentation_models as sm



#X_train, X_valid, y_train, y_valid = sk.model_selection.train_test_split(X_seg,y_seg,test_size=0.33, shuffle=True)



BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)


# In[30]:


sm.set_framework('tf.keras')
sm.framework()
#model_sm2 = sm.PSPNet(BACKBONE, encoder_weights='imagenet',classes=20, activation='softmax')


# In[31]:


dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

optim = tf.keras.optimizers.Adam(0.0001)

metrics = [UpdatedMeanIoU(num_classes=21), sm.metrics.FScore(threshold=0.5),'sparse_categorical_accuracy']

# In[41]:

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model2 = DeeplabV3Plus(image_size=512, num_classes=21)
    loss = keras.losses.SparseCategoricalCrossentropy()
    model2.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=metrics,
    )


# In[ ]:

for i in range(100):
    train_dataset = input_fn(True,'/home/arpit_manu6/dataset', 8, model2, 4)
    val_set = input_fn(False,'/home/arpit_manu6/dataset', 8, model2, 4)
    model2.fit(train_dataset,validation_data=val_set,epochs=100,steps_per_epoch=1,validation_steps=1,verbose=2)
