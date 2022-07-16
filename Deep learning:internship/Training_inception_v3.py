# Databricks notebook source
#============== TRAINING CONTROL PANEL ==============#

# ~~~~~ SELECT EXPERIMENT ~~~~~~~~~~
SELECTED_GROUP = 'Tumor' 
ALTERATION = 'immune status: hot/cold/excluded'

# ~~~~~ TRAINING PARAMS ~~~~~~~~~~
RANDOM_SEED = 42
VAL_SPLIT = 0.2
EPOCHS = 10
BATCH_SZ = 32
LR = 0.0001
# ~~~~~ NAMING ~~~~~
MODEL_NAME = 'hot_cold_path'
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
from tensorflow.keras.optimizers import SGD
import tensorflow

import mlflow
import mlflow.tensorflow as mlflow_tf

import tensorflow as tf
import multiprocessing
import random
import shutil


# COMMAND ----------

# Check TF version, check that GPU is available
print('Using tf version: {}'.format(tensorflow.version.VERSION))
print('Num CPUs available: {}'.format(multiprocessing.cpu_count()))
print('Num GPUs available: {}'.format(len(tensorflow.config.experimental.list_physical_devices('GPU'))))

# COMMAND ----------

# TRF / Image + Gene matrix metadata for this training
# We will concatenate patch fullpaths w/ alts to generate dataframes for tensorflow training
luad_res_data_path = '/dbfs/projects/immune_infiltration/data/fulldata_path_Training_2class.csv'
full_patch_path_df= pd.read_csv(luad_res_data_path, index_col = 0)

# COMMAND ----------

# Take a look at this metadata df
full_patch_path_df.head(10)

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

train_df = full_patch_path_df[full_patch_path_df['file_name_on_disk'].isin(train_hot_ids + train_cold_ids)]
val_df = full_patch_path_df[full_patch_path_df['file_name_on_disk'].isin(val_hot_ids + val_cold_ids)]
len(train_df)

# COMMAND ----------

data_path = '/dbfs/projects/immune_infiltration/data/valdata_path.csv' # May2020
val_df= pd.read_csv(data_path, index_col = 0)

#train_df = full_patch_path_df.drop(val_df.index)
#train_df = train_df.sample(frac=1)
val_df = val_df.sample(frac=1)
val_df.head(6)
val_list = val_df["file_name_on_disk"].tolist()
val_list=np. unique(val_list)
train_df= full_patch_path_df[~full_patch_path_df.file_name_on_disk.isin(val_list)]
#train_df=full_patch_path_df.drop(full_patch_path_df.specimenName.isin(val_list))

train_df = train_df.sample(frac=1)
val_df =val_df.sample(frac=1)
train_df.count
#val_list
#train_list =train_df ["specimenName"].tolist()
#train_list =np.unique(train_list )
#val_list
#train_list
#val_list

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


def augment_image(img, label):
  '''
  tensorflow image augmentations
  '''
  img = tensorflow.image.random_flip_left_right(img)
  img = tensorflow.image.random_flip_up_down(img)
  img = tensorflow.image.rot90(img)
  img = tensorflow.image.random_hue(img, max_delta=0.1)
  img = tensorflow.image.random_saturation(img, lower=0.8, upper=1.1)
  img = tensorflow.image.random_contrast(img, lower=0.9, upper=1.05)
  return img, label


def preprocess_for_inception(img, label):
  '''
  preprocessing fxn in the [-1, 1] range expected by inception
  '''
  img_uint8 = tensorflow.image.convert_image_dtype(img, dtype=tensorflow.uint8)
  
  img_uint8 = img_uint8/255
  img_uint8 = img_uint8 - 0.5
  img_uint8 = img_uint8*2
  
  return img_uint8, label

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

# TRAIN SET WITH AUGMENTATION
#strategy = tf.distribute.MirroredStrategy()
#BUFFER_SIZE = 10000
#BATCH_SIZE_PER_REPLICA = 64
#BATCH_SZ = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
#LR  =  LR * strategy.num_replicas_in_sync
train_dataset_aug = tensorflow.data.Dataset.from_tensor_slices((train_fullpaths, train_labels))

train_dataset_aug = train_dataset_aug.shuffle(len(train_fullpaths), seed=RANDOM_SEED)
train_dataset_aug = train_dataset_aug.map(lambda filename, label: parse_and_resize_image(filename, label, 448), num_parallel_calls=32)
train_dataset_aug = train_dataset_aug.map(augment_image, num_parallel_calls=32)

# will expect img to be in float32 form, need to convert for inception preprocessing
train_dataset_aug = train_dataset_aug.map(lambda filename, label: preprocess_for_inception(filename, label))
train_dataset_aug = train_dataset_aug.batch(BATCH_SZ)

train_dataset_aug = train_dataset_aug.prefetch(1)# pre-fetch 1 batch in advance of what is being passed through



# COMMAND ----------

# VAL SET (note no aug here)
val_dataset = tensorflow.data.Dataset.from_tensor_slices((val_fullpaths, val_labels))
val_dataset = val_dataset.shuffle(len(val_fullpaths), seed=RANDOM_SEED)
val_dataset = val_dataset.map(lambda filename, label: parse_and_resize_image(filename, label, 448), num_parallel_calls=32)
val_dataset = val_dataset.map(lambda filename, label: preprocess_for_inception(filename, label))
val_dataset = val_dataset.batch(BATCH_SZ)
val_dataset = val_dataset.prefetch(1)

# COMMAND ----------

# static training settings for this particular run
# for example, won't always use L2 norm as a form of regularizer

L2_REG = 0.075
OPTIMIZER = tensorflow.keras.optimizers.RMSprop(learning_rate=LR)

# directories to save model weight checkpoints and model histories
CHECKPOINT_DIR = '/dbfs/projects/immune_infiltration/data/modeling/modelWS2_weights/'
HISTORY_DIR = '/dbfs/projects/immune_infiltration/data/modeling/WS2histories/'
model_checkpoint_dir = CHECKPOINT_DIR

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model setup, callbacks
# MAGIC 
# MAGIC We'll use inception V3 as the backbone here with dropout and L2 norm for regularization. Tensorflow and tf.keras have tons of great documentation for many more possible things you can do, this is designed as one particular example.

# COMMAND ----------

l2_reg = regularizers.l2(L2_REG)

# Instantiate the model
# create the base pre-trained model
#v3 = inception_v3.InceptionV3(weights='imagenet', include_top=False, classes=3)

# add a global spatial average pooling layer
#x = v3.output
#x = layers.GlobalAveragePooling2D()(v3.output)
#x = layers.Dropout(rate=0.3)(x) #droput 
#x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
#x = layers.Dense(1024, activation='relu')(x)

#preds = layers.Dense(3, activation='softmax')(x)
#test_v3 = models.Model(v3.inputs, preds)
# this is the model we will train


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
    #layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#test_v3.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
strategy = tf.distribute.MirroredStrategy()


#from tensorflow.keras.optimizers import SGD
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
#strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  v3 = inception_v3.InceptionV3(include_top=False) # bring in Inception w/out default output nodes
  x = layers.GlobalAveragePooling2D()(v3.output)
  x = layers.Dropout(rate=0.5)(x) #droput 
  preds = layers.Dense(1, activation='sigmoid', activity_regularizer=l2_reg)(x)
  test_v3 = models.Model(v3.inputs, preds)
  for layer in test_v3.layers:
    layer.trainable = True
  test_v3.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])



#l2_reg = regularizers.l2(L2_REG)
#v3 = inception_v3.InceptionV3(include_top=False) # bring in Inception w/out default output nodes
#x = layers.GlobalAveragePooling2D()(v3.output)
#x = layers.Dropout(rate=0.5)(x) #droput 
#x = layers.Dense(1024, activation='relu', activity_regularizer=l2_reg)(x)
#preds = layers.Dense(1, activation='sigmoid', activity_regularizer=l2_reg)(x)
#test_v3 = models.Model(v3.inputs, preds)

#test_v3.compile(loss=loss, metrics=['accuracy'], optimizer='adam')

# Set up some callbacks
#model_checkpoint_dir = os.path.join(CHECKPOINT_DIR, MODEL_NAME)


# COMMAND ----------

model_checkpoint_dir = CHECKPOINT_DIR
if not os.path.isdir(model_checkpoint_dir): os.mkdir(model_checkpoint_dir) # make sure the checkpoint dir exists
  
# checkpoint to only save weights if val acc has improved since previous best

####################
# NOTE: AS OF THIS WRITING, THERE IS FREQUENTLY A DATABRICKS MEMORY PROBLEM WHEN TRYING TO WRITE HDF5 CHECKPOINTS DIRECTLY TO /dbfs/ 
# One possible workaround at this time is to save checkpoint weights to tmp in the current node, and later transfer those to desired location in dbfs.
# For this reason, we save to tmp in this notebook instead of a more straightforward checkpoint command like: 
# model_checkpointer = ModelCheckpoint(model_checkpoint_dir + '/{}_epoch'.format(MODEL_NAME) + '{epoch:02d}.hdf5', monitor='val_accuracy', save_best_only=True)
####################
#model_checkpointer = ModelCheckpoint('/tmp23' + '/{}_epoch'.format(MODEL_NAME) + '{epoch:02d}.hdf5', monitor='val_accuracy',     save_best_only=True)
#csv_logger = CSVLogger(HISTORY_DIR + MODEL_NAME + '.csv', append=True)

# reduce learning rate if val loss stops improving
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              #patience=2, min_lr=0.0001, verbose=1)
#callbacks = [model_checkpointer, csv_logger, reduce_lr]


# COMMAND ----------



# COMMAND ----------

#test_v3.fit(train_dataset_aug, validation_data=val_dataset, verbose=1, epochs=EPOCHS, callbacks=callbacks) # verbose = 1 to track progress

# COMMAND ----------

#for i, layer in enumerate(v3.layers):
   #print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
#for layer in test_v3.layers[:-10]:
  #layer.trainable = False
#for layer in test_v3.layers[-10:]:
  #layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

#test_v3.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
#test_v3.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model_checkpointer = ModelCheckpoint('/tmp' + '/{}_epoch'.format(MODEL_NAME) + '{epoch:02d}.hdf5', monitor='val_accuracy',     save_best_only=True)
csv_logger = CSVLogger(HISTORY_DIR + MODEL_NAME + '.csv', append=True)

# reduce learning rate if val loss stops improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.0001, verbose=1)
callbacks = [model_checkpointer, csv_logger, reduce_lr]

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
test_v3.fit(train_dataset_aug, validation_data=val_dataset, verbose=1, epochs=EPOCHS,callbacks=callbacks)

# COMMAND ----------

# TRAIN AND LOG
#with mlflow.start_run() as run:
  
  #mlflow.log_param('model name', MODEL_NAME)
  # mlflow.log_param('selected group', SELECTED_GROUP) # two-stage modeling, not used in this particular run!
  #mlflow.log_param('alteration target', ALTERATION)
  #mlflow.log_param('patch / FOV approach', 'resize 1024 to 448')
  #mlflow.log_param('input dims', '448')
  #mlflow.log_param('initial learning rate', LR)
  #mlflow.log_param('model architecture', 'inceptionV3, global max pool to sigmoid')
  #mlflow.log_param('shuffle seed', RANDOM_SEED)
  #mlflow.log_param('epochs', EPOCHS)
  #mlflow.log_param('num gpus', len(tensorflow.config.experimental.list_physical_devices('GPU')))
  #mlflow.log_param('total batch size', BATCH_SZ)
  #mlflow.log_param('notes', 'comp path onboarding model, EGFR soft-label using 10/10 mut/wt small resection specimens')
  # mlflow.log_param('upstream categorical model', 'categorical_010420_0033_epoch01.hdf5') # two-stage modeling, not used in this particular run!
  #mlflow.log_param('val split', VAL_SPLIT)

  
  #test_v3.fit(train_dataset_aug, validation_data=val_dataset, verbose=1, epochs=EPOCHS, callbacks=callbacks) # verbose = 1 to track progress

# COMMAND ----------

# MAGIC %md Move weight hdf5 to the checkpoint directory corresponding with this NB.

# COMMAND ----------

for weight_paths in glob.glob('/tmp/*hdf5'):
  print(weight_paths)
  shutil.move(weight_paths, model_checkpoint_dir + '/' + weight_paths.split('/')[-1])

# COMMAND ----------

# MAGIC %md
# MAGIC Plot corresponding accuracies, losses - this information can also be found later in the history file.

# COMMAND ----------

fig, (ax0, ax1) = plt.subplots(1,2, figsize=(10,4))

ax0.plot(test_v3.history.history['accuracy'],'o--')
ax0.plot(test_v3.history.history['val_accuracy'],'o--')
ax0.legend(['Accuracy', 'Val Accuracy'])
ax0.set_title('Training accuracy')
ax0.set_xlabel('Epoch')
ax0.set_ylabel('Patch Accuracy')

ax1.plot(test_v3.history.history['loss'],'o--')
ax1.plot(test_v3.history.history['val_loss'],'o--')
ax1.legend(['Loss', 'Val Loss'])
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

display()
plt.close()

# COMMAND ----------





