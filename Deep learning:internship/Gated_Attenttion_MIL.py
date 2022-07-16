# Databricks notebook source
#============== TRAINING CONTROL PANEL ==============#

# ~~~~~ SELECT EXPERIMENT ~~~~~~~~~~
SELECTED_GROUP = 'Tumor' 
ALTERATION = 'immune status: hot/cold/excluded'
# ~~~~~ TRAINING PARAMS ~~~~~~~~~~
RANDOM_SEED = 42
VAL_SPLIT = 0.2
EPOCHS = 30
BATCH_SZ = 32
LR = 0.001
BATCH_SIZE = 2 
PATCHES_PER_BAG = 50
EPOCHS = 10
LR = 0.000001
PATCH_SIZE = 224
VAL_PATCHES = 50
VAL_PER_N_EPOCHS = 1 
# ~~~~~ NAMING ~~~~~
MODEL_NAME = 'hot/cold_immune'
print('Running training for: {}'.format(MODEL_NAME))

# COMMAND ----------

# Import libraries
import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

import tensorflow

import mlflow
import mlflow.tensorflow as mlflow_tf

import tensorflow as tf
import multiprocessing
import random
import shutil
from sklearn.model_selection import KFold

# COMMAND ----------

# Check TF version, check that GPU is available
print('Using tf version: {}'.format(tensorflow.version.VERSION))
print('Num CPUs available: {}'.format(multiprocessing.cpu_count()))
print('Num GPUs available: {}'.format(len(tensorflow.config.experimental.list_physical_devices('GPU'))))

# COMMAND ----------

# TRF / Image + Gene matrix metadata for this training
# We will concatenate patch fullpaths w/ alts to generate dataframes for tensorflow training
luad_res_data_path = '/dbfs/projects/immune_infiltration/data/Tumor_and_immune_path_Training_2class.csv' # May2020
full_patch_path_df= pd.read_csv(luad_res_data_path, index_col = 0)

# COMMAND ----------

# Take a look at this metadata df
full_patch_path_df.head(10)
full_patch_path_df = full_patch_path_df.sample(frac=1, random_state=42)

# COMMAND ----------

# final structure


# COMMAND ----------

print('total patches in dataset: {}'.format(len(full_patch_path_df)))
print('Immune cold patches in dataset: {}'.format(len(full_patch_path_df[full_patch_path_df[ALTERATION]==0])))
print('Immune hot type patches in dataset: {}'.format(len(full_patch_path_df[full_patch_path_df[ALTERATION]==1])))
print('Excluded type patches in dataset: {}'.format(len(full_patch_path_df[full_patch_path_df[ALTERATION]==2])))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Train/val split

# COMMAND ----------

random.seed(RANDOM_SEED) # set random seed for reproducibility

# get unique whole-slides
cold_img_ids = list(set(full_patch_path_df[full_patch_path_df[ALTERATION]==0]['file_name_on_disk']))
hot_img_ids = list(set(full_patch_path_df[full_patch_path_df[ALTERATION]==1]['file_name_on_disk']))
excluded_img_ids = list(set(full_patch_path_df[full_patch_path_df[ALTERATION]==2]['file_name_on_disk']))

# get number to split for val
n_val_cold = int(VAL_SPLIT*len(cold_img_ids)) 
n_val_hot = int(VAL_SPLIT*len(hot_img_ids))
n_val_ex = int(VAL_SPLIT*len(excluded_img_ids))
for train_index,test_index in KFold(5).split(cold_img_ids):
  print(train_index)
len(cold_img_ids  )
  

# COMMAND ----------

# shuffle the slides and select a random val_split from them for val set.
random.shuffle(cold_img_ids)
random.shuffle(hot_img_ids)
random.shuffle(excluded_img_ids)

train_cold_ids = cold_img_ids[:-n_val_cold]
val_cold_ids = cold_img_ids[-n_val_cold:]

train_hot_ids = hot_img_ids[:-n_val_hot]
val_hot_ids = hot_img_ids[-n_val_hot:]

train_ex_ids = excluded_img_ids[:-n_val_ex]
val_ex_ids = excluded_img_ids[-n_val_ex:]


# COMMAND ----------

train_df = full_patch_path_df[full_patch_path_df['file_name_on_disk'].isin(train_hot_ids + train_cold_ids+ train_ex_ids)]
val_df = full_patch_path_df[full_patch_path_df['file_name_on_disk'].isin(val_hot_ids + val_cold_ids+val_ex_ids)]
len(train_df)

# COMMAND ----------

train_df = train_df.sample(frac=1,random_state=42)
val_df = val_df.sample(frac=1,random_state=42 )
val_df.to_csv('valdata_MILATT_path.csv')




# COMMAND ----------

# MAGIC %sh cp valdata_MILATT_path.csv /dbfs/projects/immune_infiltration/data/

# COMMAND ----------

import sys
import time
from random import shuffle
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow import keras
#from keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply
#from .metrics import bag_accuracy, bag_loss
#from .custom_layers import Mil_Attention, Last_Sigmoid
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers
from tensorflow.keras.applications import ResNet50
l2_penalty = 0.01
l2_reg = tf.keras.regularizers.L2(l2_penalty)

class Mil_Attention(Layer):
    """
    Mil Attention Mechanism

    This layer contains Mil Attention Mechanism

    # Input Shape
        2D tensor with shape: (batch_size, input_dim)

    # Output Shape
        2D tensor with shape: (1, units)
    """

    def __init__(self, L_dim, output_dim, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                    use_bias=True, use_gated=False, **kwargs):
        self.L_dim = L_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.use_gated = use_gated

        self.v_init = initializers.get(kernel_initializer)
        self.w_init = initializers.get(kernel_initializer)
        self.u_init = initializers.get(kernel_initializer)


        self.v_regularizer = regularizers.get(kernel_regularizer)
        self.w_regularizer = regularizers.get(kernel_regularizer)
        self.u_regularizer = regularizers.get(kernel_regularizer)

        super(Mil_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.V = self.add_weight(shape=(input_dim, self.L_dim),
                                      initializer=self.v_init,
                                      name='v',
                                      regularizer=self.v_regularizer,
                                      trainable=True)


        self.w = self.add_weight(shape=(self.L_dim, 1),
                                    initializer=self.w_init,
                                    name='w',
                                    regularizer=self.w_regularizer,
                                    trainable=True)


        if self.use_gated:
            self.U = self.add_weight(shape=(input_dim, self.L_dim),
                                     initializer=self.u_init,
                                     name='U',
                                     regularizer=self.u_regularizer,
                                     trainable=True)
        else:
            self.U = None

        self.input_built = True


    def call(self, x, mask=None):
        n, d = x.shape
        ori_x = x
        # do Vhk^T
        x = K.tanh(K.dot(x, self.V)) # (2,64)

        if self.use_gated:
            gate_x = K.sigmoid(K.dot(ori_x, self.U))
            ac_x = x * gate_x
        else:
            ac_x = x

        # do w^T x
        soft_x = K.dot(ac_x, self.w)  # (2,64) * (64, 1) = (2,1)
        alpha = K.softmax(K.transpose(soft_x)) # (2,1)
        alpha = K.transpose(alpha)
        return alpha

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'v_initializer': initializers.serialize(self.V.initializer),
            'w_initializer': initializers.serialize(self.w.initializer),
            'v_regularizer': regularizers.serialize(self.v_regularizer),
            'w_regularizer': regularizers.serialize(self.w_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Mil_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Last_Sigmoid(Layer):
    """
    Attention Activation
    This layer contains a FC layer which only has one neural with sigmoid actiavtion
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    This layer is used in mi-Net.

    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)

    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Last_Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape
        x = K.sum(x, axis=0, keepdims=True)
        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        out = K.sigmoid(x)


        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Last_Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def bag_accuracy(y_true, y_pred):
    """Compute accuracy of one bag.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    acc : Tensor (1 x 1)
        Accuracy of bag label prediction.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    acc = K.mean(K.equal(y_true, K.round(y_pred)))
    return acc


def bag_loss(y_true, y_pred):
    """Compute binary crossentropy loss of predicting bag loss.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    acc : Tensor (1 x 1)
        Binary Crossentropy loss of predicting bag label.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss

def mil_net(input_dim, useMulGpu=False):

    #lr = args.init_lr
    #weight_decay = args.init_lr
    #momentum = args.momentum
    
    lr = .0001
    weight_decay = .9
    momentum = .0005
    #data_input = Input(shape=input_dim, dtype='float32', name='input')
    #conv1 = Conv2D(36, kernel_size=(4,4), kernel_regularizer=l2(weight_decay), activation='relu')(data_input)
    #conv1 = MaxPooling2D((2,2))(conv1)

    #conv2 = Conv2D(48, kernel_size=(3,3),  kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    #conv2 = MaxPooling2D((2,2))(conv2)
    #x = Flatten()(conv2)

    #fc1 = Dense(512, activation='relu',kernel_regularizer=l2(weight_decay), name='fc1')(x)
    #fc1 = Dropout(0.5)(fc1)
    #fc2 = Dense(512, activation='relu', kernel_regularizer=l2(weight_decay), name='fc2')(fc1)
    #fc2 = Dropout(0.5)(fc2)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)

    resnet = ResNet50(input_shape=(input_dim, input_dim, 3, ))  
    resnet.layers[-1].activation = None
  
#   resnet_pool = layers.GlobalAveragePooling2D(name='global_avg_pool')(resnet.output)
    fc_1 = layers.Dense(256, activation='relu', name='embedding_reduce_1', kernel_regularizer=l2_reg)(resnet.output)
    fc_2 = layers.Dropout(0.5)(fc_1)
    reduced_embeddings = layers.Dense(128, activation='relu', name='embedding_reduce_2', kernel_regularizer=l2_reg)(fc_2)

    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=True)(fc_2)
    x_mul = multiply([alpha, fc_2])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)
    #
    model = Model(inputs=resnet.inputs, outputs=[out])

    # model.summary()

    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=bag_loss, metrics=[bag_accuracy])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=bag_loss, metrics=[bag_accuracy])
        parallel_model = model

    return parallel_model



def mil_resnet(input_dim=224):  
  


  resnet = ResNet50(input_shape=(input_dim, input_dim, 3, ))  
  resnet.layers[-1].activation = None
  
  fc_1 = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(resnet.output)
  fc_1 = layers.Dropout(0.5)(fc_1)
  reduced_embeddings = layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)(fc_1)
  #attention component
  attention_layer = layers.Dense(128, activation='tanh', kernel_regularizer=l2_reg)(reduced_embeddings)
  attention_layer = layers.Dense(1, activation=None, kernel_regularizer=l2_reg)(attention_layer)
  attention_layer = tf.transpose(attention_layer)
  att_weights = layers.Softmax(name='attention_softmax')(attention_layer)
  #classification component
  aggregated_embeddings = tf.matmul(att_weights, reduced_embeddings, name='dot_product')
  
  output = layers.Dense(1, activation='sigmoid', name='classification_linear', kernel_regularizer=l2_reg)(aggregated_embeddings)
  
  model = models.Model(inputs=resnet.inputs, outputs=output)
  
  return model

 


def mil_inception(input_dim=224):  
  
  #l2 reg
  incep = inception_v3.InceptionV3(include_top=False) 

  incep.layers[-1].activation = None
  x = layers.GlobalAveragePooling2D()(incep.output)
  fc1 = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(x)
  fc1 = layers.Dropout(0.5)(fc1)
  fc2= layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)(fc1)
  attention_layer = layers.Dense(128, activation='tanh', kernel_regularizer=l2_reg)(fc2)
  attention_layer = layers.Dense(1, activation=None, kernel_regularizer=l2_reg)(attention_layer)
  attention_layer = tf.transpose(attention_layer)
  att_weights = layers.Softmax(name='attention_softmax')(attention_layer)
  aggregated_embeddings = tf.matmul(att_weights, fc2)
  output = layers.Dense(1, activation='sigmoid', name='classification_linear', kernel_regularizer=l2_reg)(aggregated_embeddings)
  
  model = models.Model(inputs=incep.inputs, outputs=output)
  for layer in model.layers:
    layer.trainable = True
  
  return model

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Preparing tensorflow datasets, augmentations, preprocessing, etc.

# COMMAND ----------

def parse_and_resize_image(filename, label, resize_sz):
  image_string = tensorflow.io.read_file(filename)
  
  # Don't use tf.image.decode_image, or the output shape will be undefined
  image = tensorflow.io.decode_png(image_string, channels=3)
  image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
  
  # resize image
  resized_image = tensorflow.image.resize(image, [resize_sz, resize_sz]) 
  return resized_image, label


def preprocess_for_inception(img, label):
  '''
  preprocessing fxn in the [-1, 1] range expected by inception
  '''
  img_uint8 = tensorflow.image.convert_image_dtype(img, dtype=tensorflow.uint8)
  
  img_uint8 = img_uint8/255
  img_uint8 = img_uint8 - 0.5
  img_uint8 = img_uint8*2
  
  return img_uint8, label
def parse_and_resize_image(filename, resize_sz):
    #load and resize the image from full size to the given resize without cropping  
    image_string = tf.io.read_file(filename)
    image = tf.io.decode_png(image_string, channels=3)
#     image = tf.image.convert_image_dtype(image, tf.float32)
    resized_image = tf.image.resize(image, [resize_sz, resize_sz])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return resized_image
def preprocess_tanh_range(image):
    #preprocesses image data to the range [-1, 1]
    img_uint8 = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    img_uint8 = img_uint8/255
    img_uint8 = img_uint8 - 0.5
    img_uint8 = img_uint8*2
    return img_uint8 
  
def augment_image(img):
  #optionally perform random image augmentation 
  img = tf.image.random_flip_left_right(img)
  img = tf.image.random_flip_up_down(img)
  img = tf.image.rot90(img, k=np.random.randint(0,4))
  img = tf.image.random_hue(img, max_delta=0.1)
  img = tf.image.random_saturation(img, lower=0.8, upper=1.1)
  img = tf.image.random_contrast(img, lower=0.9, upper=1.05)
  return img  

# COMMAND ----------

# get training, val paths and labels to feed into tf datasets
train_fullpaths = list(train_df['fullpath'])
train_labels = list(train_df[ALTERATION])

val_fullpaths = list(val_df['fullpath'])
val_labels = list(val_df[ALTERATION])
#depth = tf.constant(3)
#train_labels = tf.one_hot(train_labels, depth=depth)
#val_labels = tf.one_hot(val_labels, depth=depth)
#train_labels=to_categorical(y_train)
#val_tables=to_categorical(y_val)
len(train_fullpaths)

# COMMAND ----------

# static training settings for this particular run
# for example, won't always use L2 norm as a form of regularizer

L2_REG = 0.01
OPTIMIZER = tensorflow.keras.optimizers.RMSprop(learning_rate=LR)

# directories to save model weight checkpoints and model histories
checkpoint_dir = '/dbfs/projects/immune_infiltration/data/modeling/modelGatedAttMIL2_weights/'

#model_checkpoint_dir = CHECKPOINT_DIR

# COMMAND ----------

import time
# ======= TRAINING LOOP =======
# set up model and optimizer
model = mil_inception(input_dim=PATCH_SIZE)
preprocess_fn = tf.keras.applications.resnet.preprocess_input
loss_obj = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.RMSprop(LR)
bin_accuracy_tracker = tf.keras.metrics.BinaryAccuracy()
auc_tracker = tf.keras.metrics.AUC()
NUM_TRAIN_BATCHES = int(np.ceil(len(train_df)/BATCH_SIZE))
best_val_auc=0

for epoch_idx in range(EPOCHS):
  start = time.time()
  epoch_total_loss = 0.0
  
  #optionally shuffle the training set each time
  train_df = train_df.sample(frac=1)
  
  #loop through each batch and perform the train step
  for batch_idx in range(NUM_TRAIN_BATCHES):
    if batch_idx <= NUM_TRAIN_BATCHES - 1:
      train_df_batch = train_df.iloc[batch_idx*BATCH_SIZE : batch_idx*BATCH_SIZE + BATCH_SIZE]
    else:
      train_df_batch = train_df.iloc[batch_idx*BATCH_SIZE:]

  
    #loop through each slide in the batch and get slide prediction 
    with tf.GradientTape() as tp:
      seen_first_preds = False    
      for i, row in train_df_batch.iterrows():
        trf = row['specimenName']              
        train_patch_df = train_df[train_df['specimenName']==trf]        
        if len(train_patch_df) < PATCHES_PER_BAG:
          selected_patch_paths = train_patch_df['fullpath']
        else:
          selected_patch_paths = train_patch_df.sample(n=PATCHES_PER_BAG)['fullpath']   
        train_patch_ds = tf.data.Dataset.from_tensor_slices(selected_patch_paths)
        train_patch_ds = train_patch_ds.map(lambda filename: parse_and_resize_image(filename, PATCH_SIZE), num_parallel_calls=8) 
        train_patch_ds = train_patch_ds.map(lambda image: augment_image(image), num_parallel_calls=8)
        train_patch_ds = train_patch_ds.map(preprocess_fn, num_parallel_calls=8)
        train_patch_ds = train_patch_ds.batch(len(selected_patch_paths))
        patch_set = next(train_patch_ds.as_numpy_iterator()) #does this slow things down, by converting to numpy iterator

        output = model(patch_set, training=True)
        
      #
        if not seen_first_preds:
            preds = output
            seen_first_preds = True
        else:
            preds = tf.concat([preds, output], axis=0)
          
      labels = train_df_batch[ALTERATION].to_numpy().astype(np.float32)
      labels_tf = tf.convert_to_tensor(labels)
      preds_tf = tf.squeeze(preds)
      bin_accuracy_tracker.update_state(labels_tf.numpy(), preds_tf.numpy())
      batch_loss = loss_obj(labels_tf, preds_tf)
      epoch_total_loss += batch_loss
      grads = tp.gradient(batch_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      acc_metrics = tf.keras.metrics.BinaryAccuracy()
      acc = acc_metrics(labels_tf, preds_tf)
      print('   epoch {} batch: {}/{}, loss: {}, acc: {}'.format(epoch_idx, batch_idx, NUM_TRAIN_BATCHES, batch_loss, acc))
      print(f'Training accuracy for each batch: {bin_accuracy_tracker.result()}\n')
    
  end = time.time()

  print(f'epoch train accuracy: {bin_accuracy_tracker.result()}')
  epoch_train_loss = epoch_total_loss/NUM_TRAIN_BATCHES
  print(f'epoch train loss: {epoch_train_loss}')
  bin_accuracy_tracker.reset_states()  
  if epoch_idx % VAL_PER_N_EPOCHS == 0:
    print(f'epoch {epoch_idx}. performing validation')
    epoch_seen_best_val_auc = -1
    seen_first_valpreds = False
    for i, row in val_df.iterrows():
      val_trf = row['specimenName']
      val_patch_df = val_df[val_df['specimenName']==val_trf]
      if len(val_patch_df) == 0:
          print(f'val trf{val_trf} has {len(val_patch_df)} patchesskipping')
          continue

      if len(val_patch_df) < VAL_PATCHES:
          val_selected_patch_paths = val_patch_df['fullpath'] 
      else:
          val_selected_patch_paths = val_patch_df.sample(n=VAL_PATCHES)['fullpath']
      val_patch_ds = tf.data.Dataset.from_tensor_slices(val_selected_patch_paths)
      val_patch_ds = val_patch_ds.map(lambda filename: parse_and_resize_image(filename, PATCH_SIZE), num_parallel_calls=8) 
      val_patch_ds = val_patch_ds.map(preprocess_fn, num_parallel_calls=8)
      val_patch_ds = val_patch_ds.batch(len(val_selected_patch_paths))
      val_patch_set = next(val_patch_ds.as_numpy_iterator()) #

      val_output = model(val_patch_set, training=True)
      if not seen_first_valpreds:
        val_preds = val_output
        seen_first_valpreds = True
      else:
        val_preds = tf.concat([val_preds, val_output], axis=0)
    val_labels = val_df[ALTERATION].to_numpy().astype(np.float32)
    val_labels_tf = tf.convert_to_tensor(val_labels)
    val_preds_tf = tf.squeeze(val_preds)
    val_auc_tracker = tf.keras.metrics.AUC()
    val_acc_tracker = tf.keras.metrics.BinaryAccuracy()
    val_auc = val_auc_tracker(val_labels_tf, val_preds_tf)
    val_acc = val_acc_tracker(val_labels_tf, val_preds_tf)
    print(f'epoch {epoch_idx} val auc: {val_auc}')
    print(f'epoch {epoch_idx} val accuracy: {val_acc}')
    if val_auc > best_val_auc:
      print(f'new best val auc: {val_auc}')
      print(f'prev best auc: {best_val_auc}')
      best_val_auc = val_auc
      epoch_seen_best_val_auc = epoch_idx
      model.save_weights('/tmp/mil_hot_cold_epoch{}.hdf5'.format(epoch_idx))
      print('copying weights to dbfs')
      dbutils.fs.cp('file:/tmp/mil_hot_cold_epoch{}.hdf5'.format(epoch_idx), (checkpoint_dir + 'mil_hot_cold_epoch{}.hdf5'.format(epoch_idx))[6:]) 
      print('model saved.')

# COMMAND ----------

WEIGHTS_PATH='/dbfs/projects/immune_infiltration/data/modeling/modelGatedAttMIL2_weights/mil_hot_cold_epoch0.hdf5'

#'/dbfs/nre/annotation/modeling/model_weights/tf2_luad_5class_longer_07_29_2020_01/tf2_luad_5class_longer_07_29_2020_01_epoch13.hdf5'
model = mil_net(input_dim=PATCH_SIZE)
#model = mil_resnet(input_dim=PATCH_SIZE)
model.load_weights(WEIGHTS_PATH)   

preprocess_fn = tf.keras.applications.resnet.preprocess_input

# COMMAND ----------

data_path = '/dbfs/projects/immune_infiltration/data/valdata_MILATT_path.csv' 
val_df= pd.read_csv(data_path, index_col = 0)
val_fullpaths = list(val_df['fullpath'])
val_labels = list(val_df[ALTERATION])
seen_first_valpreds = False
file_name=[]
for i, row in val_df.iterrows():
  val_trf = row['specimenName']
  val_patch_df = val_df[val_df['specimenName']==val_trf]
  patch_labels= row[ALTERATION]
  train_labels = list(val_patch_df[ALTERATION])
  my_final_list = set(train_labels)
  val_selected_patch_labels= list(my_final_list) 
  val_labels_bag = tf.convert_to_tensor(val_selected_patch_labels)


  #for i in range(4):
  if len(val_patch_df)==0:
    continue
  if len(val_patch_df) < VAL_PATCHES:
    val_selected_patch_paths = val_patch_df['fullpath']
    val_file_name = row['file_name_on_disk']
    file_name.append(val_file_name) 

   
  else:
    val_selected_patch_paths = val_patch_df.sample(n=VAL_PATCHES)['fullpath']
    val_file_name = row['file_name_on_disk']
    file_name.append(val_file_name) 
                     
  val_df.drop(val_selected_patch_paths.index, axis=0,inplace=True)

  val_patch_ds = tf.data.Dataset.from_tensor_slices(val_selected_patch_paths)
  
  val_patch_ds = val_patch_ds.map(lambda filename: parse_and_resize_image(filename, PATCH_SIZE), num_parallel_calls=8) 
  val_patch_ds = val_patch_ds.map(preprocess_fn, num_parallel_calls=8)
  val_patch_ds = val_patch_ds.batch(len(val_selected_patch_paths))
  val_patch_set = next(val_patch_ds.as_numpy_iterator()) #does this slow things down, by converting to numpy iterator

  val_output = model(val_patch_set, training=True)
  val_output2 = val_output
  l=len(val_selected_patch_paths)
  #for i in range(1,l):
    #val_output2 = tf.concat([val_output2, val_output], axis=0)
  if not seen_first_valpreds:
    val_preds = val_output
    val_labels=val_labels_bag
    seen_first_valpreds = True
  else:
    val_preds = tf.concat([val_preds, val_output], axis=0)
    val_labels = tf.concat([val_labels, val_labels_bag], axis=0)
    #val_df2.drop(val_selected_patch_paths.index, axis=0,inplace=True)
#val_labels = val_df[ALTERATION].to_numpy().astype(np.float32)
#val_labels_tf = tf.convert_to_tensor(val_labels)
#att_weights = att_model(val_patch_set)
val_preds_tf = tf.squeeze(val_preds)
val_labels_tf = tf.squeeze(val_labels)
val_auc_tracker = tf.keras.metrics.AUC()
val_acc_tracker = tf.keras.metrics.BinaryAccuracy()
val_auc = val_auc_tracker(val_labels_tf, val_preds_tf)
val_acc = val_acc_tracker(val_labels_tf, val_preds_tf)
val_preds_tf.shape

# COMMAND ----------

from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(val_labels_tf,val_preds_tf)

roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2
roc_auc=roc_auc
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
len(tpr)
fpr

# COMMAND ----------

data_path = '/dbfs/projects/immune_infiltration/data/valdata_MILATT_path.csv' 
val_df= pd.read_csv(data_path, index_col = 0)
val_fullpaths = list(val_df['fullpath'])
val_labels = list(val_df[ALTERATION])
seen_first_valpreds = False
for i, row in val_df.iterrows():
  val_trf = row['specimenName']
  val_patch_df = val_df[val_df['specimenName']==val_trf]
  if len(val_patch_df) < NUM_VAL_PATCHES:
    val_selected_patch_paths = val_patch_df['fullpath'] 
  else:
    val_selected_patch_paths = val_patch_df.sample(n=NUM_VAL_PATCHES)['fullpath']

  val_patch_ds = tf.data.Dataset.from_tensor_slices(val_selected_patch_paths)
  val_patch_ds = val_patch_ds.map(lambda filename: parse_and_resize_image(filename, PATCH_SIZE), num_parallel_calls=8) 
  val_patch_ds = val_patch_ds.map(preprocess_fn, num_parallel_calls=8)
  val_patch_ds = val_patch_ds.batch(len(val_selected_patch_paths))
  val_patch_set = next(val_patch_ds.as_numpy_iterator()) #does this slow things down, by converting to numpy iterator

  val_output = model(val_patch_set, training=True)
  if not seen_first_valpreds:
    val_preds = val_output
    seen_first_valpreds = True
  else:
    val_preds = tf.concat([val_preds, val_output], axis=0)
val_labels = val_df[ALTERATION].to_numpy().astype(np.float32)
val_labels_tf = tf.convert_to_tensor(val_labels)
val_preds_tf = tf.squeeze(val_preds)
val_auc_tracker = tf.keras.metrics.AUC()
val_acc_tracker = tf.keras.metrics.BinaryAccuracy()
val_auc = val_auc_tracker(val_labels_tf, val_preds_tf)
val_acc = val_acc_tracker(val_labels_tf, val_preds_tf)
val_preds_tf.shape
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(val_labels_tf,val_preds_tf)

roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2
roc_auc=roc_auc
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
len(tpr)
fpr

# COMMAND ----------

file_name= list(val_df['file_name_on_disk'])
y_pred_keras2 =list(np.array(val_preds_tf))
y_pred_keras2
y_pred_keras2 = [float(i) for i in y_pred_keras2]

df = pd.DataFrame(
    {'y_pred': y_pred_keras2,
     'val_labels':val_labels_tf ,
     'file_name_on_disk': file_name
    })
df
slice_pred=df.groupby('file_name_on_disk', as_index=False)['y_pred'].mean()
slice_label=df.groupby('file_name_on_disk', as_index=False)['val_labels'].mean()
slice_y_pred_keras2= list(slice_pred['y_pred'])

slice_val_labels= list(slice_label['val_labels'])

# COMMAND ----------

# MAGIC %md
# MAGIC Plot corresponding accuracies, losses - this information can also be found later in the history file.

# COMMAND ----------


from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve( slice_val_labels,slice_y_pred_keras2)

roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2

plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

