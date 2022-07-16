# Databricks notebook source
# MAGIC %md
# MAGIC # Computational pathology ML onboarding
# MAGIC 
# MAGIC This set of onboarding materials walks through an example problem to familiarize the user with the computational pathology pipeline and workflow. These notebooks serve as starter kits and recipes for future analyses and should not be treated as the most highly optimized ways to push for state of the art results at scale. The use case we will analyze here is EGFR alteration prediction from whole-slide images. We have pre-selected a small set of sample data to make some of these steps computationally tractible in a shorter timeframe, particularly for notebook 2 (training)!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notebook 2 - training pipeline
# MAGIC 
# MAGIC The objective of this notebook is to provide an introductory example of how patch sets from whole-slide H&E images are run through deep learning algorithms to train models. We will go through the steps involved in saving . This pipeline contains GPU-intensive operations, and should be run on a **GPU cluster**. GPU_p2xl_01 has all required libraries etc and can serve as a starting point here.
# MAGIC 
# MAGIC Our current training pipelines are performed through tensorflow 2 API's. **Your cluster should have dbfs:/initscripts/tensorflow21.sh** as an init script in its Advanced Options -> Init Scripts configuration settings.

# COMMAND ----------

# MAGIC %md Set up some training params

# COMMAND ----------

#============== TRAINING CONTROL PANEL ==============#

# ~~~~~ SELECT EXPERIMENT ~~~~~~~~~~
SELECTED_GROUP = 'Tumor' # we won't subset by tumor here, but do often gate by morphology in more exotic two-stage models.
ALTERATION = 'EGFR'

# ~~~~~ TRAINING PARAMS ~~~~~~~~~~
RANDOM_SEED = 13
VAL_SPLIT = 0.2
EPOCHS = 3
BATCH_SZ = 16
LR = 0.001

# ~~~~~ NAMING ~~~~~
MODEL_NAME = 'tf2_egfr_cat_06_03_2020_01'
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
luad_res_data_path = '/dbfs/data/metadata/luad_resection_cohort_w_genes_v5.xls' # May2020
luad_res_df = pd.read_excel(luad_res_data_path, index_col = 0)

# COMMAND ----------

# Take a look at this metadata df
luad_res_df.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC Build out fullpath dataframes from patch index .csv files generated in notebook 1, w/ their associated labels.

# COMMAND ----------

# patched out dataframes for this ex
patch_fullpaths_dir = '/dbfs/data/comppath-onboarding/patch_paths/'

# add gene matrix etc, concat the patch df's all together

count = 0
for patch_csvs in glob.glob(patch_fullpaths_dir + '*csv'):
  img_id = patch_csvs.split('/')[-1].split('.')[0]
  patch_df_with_joinable_key = pd.read_csv(patch_csvs, index_col = 0)
  patch_df_with_joinable_key['file_name_on_disk'] = img_id
  merged_df = pd.merge(patch_df_with_joinable_key, luad_res_df, on = 'file_name_on_disk')
  
  # we don't need to lug along all this metadata / info for every coming operation, although some could be included in a model. 
  # just keep file_name_on_disk to val split at the whole-slide level, fullpath, and in this case, EGFR.
  # this is strictly optional, slightly silly for this example where there aren't many patches and only the image as input!
  
  merged_df = merged_df[['file_name_on_disk', 'fullpath', ALTERATION]]
  
  if count == 0: # if after the first one df, concat onto existing
    full_patch_path_df = merged_df
  else:
    full_patch_path_df = pd.concat([full_patch_path_df, merged_df], ignore_index = True)
    
  count = count + 1

# COMMAND ----------

# final structure
full_patch_path_df.head(3)

# COMMAND ----------

# MAGIC %md Patch counts for total samples in set, EGFR MUT and EGFR WT counts. We split at the image level rather than the patch level, but ideally the patch numbers themselves aren't wildly imbalanced.

# COMMAND ----------

print('total patches in dataset: {}'.format(len(full_patch_path_df)))
print('EGFR mutant patches in dataset: {}'.format(len(full_patch_path_df[full_patch_path_df[ALTERATION]==1])))
print('EGFR wild type patches in dataset: {}'.format(len(full_patch_path_df[full_patch_path_df[ALTERATION]==0])))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Train/val split

# COMMAND ----------

random.seed(RANDOM_SEED) # set random seed for reproducibility

# get unique whole-slides
mut_img_ids = list(set(full_patch_path_df[full_patch_path_df[ALTERATION]==1]['file_name_on_disk']))
wt_img_ids = list(set(full_patch_path_df[full_patch_path_df[ALTERATION]==0]['file_name_on_disk']))

# get number to split for val
n_val_mut = int(VAL_SPLIT*len(mut_img_ids)) 
n_val_wt = int(VAL_SPLIT*len(wt_img_ids))

# COMMAND ----------

# shuffle the slides and select a random val_split from them for val set.
random.shuffle(mut_img_ids)
random.shuffle(wt_img_ids)

train_mut_ids = mut_img_ids[:-n_val_mut]
val_mut_ids = mut_img_ids[-n_val_mut:]

train_wt_ids = wt_img_ids[:-n_val_wt]
val_wt_ids = wt_img_ids[-n_val_wt:]

# COMMAND ----------

train_df = full_patch_path_df[full_patch_path_df['file_name_on_disk'].isin(train_mut_ids + train_wt_ids)]
val_df = full_patch_path_df[full_patch_path_df['file_name_on_disk'].isin(val_mut_ids + val_wt_ids)]

# COMMAND ----------

# MAGIC %md 
# MAGIC The next two lines in preparing for training are small but **VERY IMPORTANT**.
# MAGIC **You have to shuffle these dataframes or the training dynamics are not good** (very long string of label 0 samples followed by very long string of label 1 samples etc, bad for the gradient descent / weight updates).

# COMMAND ----------

train_df = train_df.sample(frac=1)
val_df = val_df.sample(frac=1)

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

# COMMAND ----------

# TRAIN SET WITH AUGMENTATION

train_dataset_aug = tensorflow.data.Dataset.from_tensor_slices((train_fullpaths, train_labels))
train_dataset_aug = train_dataset_aug.shuffle(len(train_fullpaths), seed=RANDOM_SEED)
train_dataset_aug = train_dataset_aug.map(lambda filename, label: parse_and_resize_image(filename, label, 448), num_parallel_calls=4)
train_dataset_aug = train_dataset_aug.map(augment_image, num_parallel_calls=4)

# will expect img to be in float32 form, need to convert for inception preprocessing
train_dataset_aug = train_dataset_aug.map(lambda filename, label: preprocess_for_inception(filename, label))
train_dataset_aug = train_dataset_aug.batch(BATCH_SZ)

train_dataset_aug = train_dataset_aug.prefetch(1) # pre-fetch 1 batch in advance of what is being passed through

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
CHECKPOINT_DIR = '/dbfs/data/comppath-onboarding/modeling/model_weights/'
HISTORY_DIR = '/dbfs/data/comppath-onboarding/modeling/histories/'


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model setup, callbacks
# MAGIC 
# MAGIC We'll use inception V3 as the backbone here with dropout and L2 norm for regularization. Tensorflow and tf.keras have tons of great documentation for many more possible things you can do, this is designed as one particular example.

# COMMAND ----------

# Instantiate the model
l2_reg = regularizers.l2(L2_REG)

v3 = inception_v3.InceptionV3(include_top=False) # bring in Inception w/out default output nodes
x = layers.GlobalAveragePooling2D()(v3.output)
x = layers.Dropout(rate=0.3)(x) #droput 
preds = layers.Dense(1, activation='sigmoid', activity_regularizer=l2_reg)(x)
test_v3 = models.Model(v3.inputs, preds)
  
test_v3.compile(loss='binary_crossentropy', metrics=['accuracy'])


# Set up some callbacks
model_checkpoint_dir = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
if not os.path.isdir(model_checkpoint_dir): os.mkdir(model_checkpoint_dir) # make sure the checkpoint dir exists
  
# checkpoint to only save weights if val acc has improved since previous best

####################
# NOTE: AS OF THIS WRITING, THERE IS FREQUENTLY A DATABRICKS MEMORY PROBLEM WHEN TRYING TO WRITE HDF5 CHECKPOINTS DIRECTLY TO /dbfs/ 
# One possible workaround at this time is to save checkpoint weights to tmp in the current node, and later transfer those to desired location in dbfs.
# For this reason, we save to tmp in this notebook instead of a more straightforward checkpoint command like: 
# model_checkpointer = ModelCheckpoint(model_checkpoint_dir + '/{}_epoch'.format(MODEL_NAME) + '{epoch:02d}.hdf5', monitor='val_accuracy', save_best_only=True)
####################

model_checkpointer = ModelCheckpoint('/tmp' + '/{}_epoch'.format(MODEL_NAME) + '{epoch:02d}.hdf5', monitor='val_accuracy', save_best_only=True)
csv_logger = CSVLogger(HISTORY_DIR + MODEL_NAME + '.csv', append=True)

# reduce learning rate if val loss stops improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.0001, verbose=1)
callbacks = [model_checkpointer, csv_logger, reduce_lr]


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Train and log
# MAGIC MLFlow provides one way to log associated parameters and metadata with a given run. Also encourage looking at the documentation there; we will wrap it around our model fit step here.

# COMMAND ----------

# TRAIN AND LOG
with mlflow.start_run() as run:
  
  mlflow.log_param('model name', MODEL_NAME)
  # mlflow.log_param('selected group', SELECTED_GROUP) # two-stage modeling, not used in this particular run!
  mlflow.log_param('alteration target', ALTERATION)
  mlflow.log_param('patch / FOV approach', 'resize 1024 to 448')
  mlflow.log_param('input dims', '448')
  mlflow.log_param('initial learning rate', LR)
  mlflow.log_param('model architecture', 'inceptionV3, global max pool to sigmoid')
  mlflow.log_param('shuffle seed', RANDOM_SEED)
  mlflow.log_param('epochs', EPOCHS)
  mlflow.log_param('num gpus', len(tensorflow.config.experimental.list_physical_devices('GPU')))
  mlflow.log_param('total batch size', BATCH_SZ)
  mlflow.log_param('notes', 'comp path onboarding model, EGFR soft-label using 10/10 mut/wt small resection specimens')
  # mlflow.log_param('upstream categorical model', 'categorical_010420_0033_epoch01.hdf5') # two-stage modeling, not used in this particular run!
  mlflow.log_param('val split', VAL_SPLIT)

  test_v3.fit(train_dataset_aug, validation_data=val_dataset, verbose=1, epochs=EPOCHS, callbacks=callbacks) # verbose = 1 to track progress

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

# MAGIC %md
# MAGIC Note that the training has not clearly saturated here, 3 epochs is not a ton! For production runs, it is advised to implement early stopping and run until the validation metric stops improving. 
# MAGIC 
# MAGIC Additionally, MLflow model control params for full runs performed in this notebook can be accessed in the top right corner via Runs -> View Run Detail. 
# MAGIC 
# MAGIC In notebook 3, we will leverage these weights to visualize the predictions on a whole slide.
