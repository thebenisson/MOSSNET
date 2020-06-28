import tensorflow as tf
import pathlib
import os
from utils import Params

params = Params()
train_dir = pathlib.Path('../data/input/mossignatures/TrainingSet')
img_count = len(list(train_dir.glob('*/*/*.png')))

CLASSNAMES = [item.name + '-' + str(item.parent.name) for item in train_dir.glob('*/*')]
LABELS = tf.convert_to_tensor(tf.constant(CLASSNAMES))
CACHE = True
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]+'-'+parts[-3]  # label is a string
    indx = tf.where(tf.equal(label,LABELS)) # get corresponding integer ID for label
    indx = tf.cast(indx, tf.float32)
    return indx

def decode_img(img):
    img = tf.image.decode_png(img, channels=params.channels)
    img = tf.image.resize(img, [params.img_width, params.img_height])
    return img/255.0

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def train_data_loader(cache=CACHE, shuffle_buffer_size=2000):
    list_ds = tf.data.Dataset.list_files(str(train_dir/'*/*/*'))
    ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(params.batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

#TODO: load data from kaggle into input path