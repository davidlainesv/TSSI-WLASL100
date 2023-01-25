import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from config import INPUT_HEIGHT, INPUT_WIDTH, RANDOM_SEED
from data_augmentation import RandomFlip, RandomScale, RandomShift, RandomRotation, RandomSpeed
from preprocessing import PadIfLessThan, ResizeIfMoreThan, preprocess_dataframe
from skeleton_graph import tssi_v2
from sklearn.preprocessing import OneHotEncoder
import numpy as np


available_augmentations = {
    'scale': RandomScale(min_value=0.0, max_value=255.0, seed=1),
    'shift': RandomShift(min_value=0.0, max_value=255.0, seed=2),
    'flip': RandomFlip("horizontal", max_value=255.0, seed=3),
    'rotation': RandomRotation(factor=15.0, min_value=0.0, max_value=255.0, seed=4),
    'speed': RandomSpeed(frames=128, seed=5)
}

augmentations_order = ['scale', 'shift', 'flip', 'rotation', 'speed']


def dataframe_to_dataset(dataframe, columns, filter_video_ids=[]):
    x_sorted_columns = [col + "_x" for col in columns]
    y_sorted_columns = [col + "_y" for col in columns]

    enc = OneHotEncoder()

    if len(filter_video_ids) > 0:
        dataframe = dataframe[dataframe["video"].isin(filter_video_ids)]

    dataframe_length = dataframe.shape[0]
    num_columns = len(x_sorted_columns)
    stacked_images = np.zeros((dataframe_length, num_columns, 3))
    stacked_images[:, :, 0] = dataframe.loc[:, x_sorted_columns].to_numpy()
    stacked_images[:, :, 1] = dataframe.loc[:, y_sorted_columns].to_numpy()
    video_labels = dataframe.groupby("video")["label"].unique().tolist()
    video_lengths = list(dataframe.groupby("video")["frame"].count())

    X = tf.RaggedTensor.from_row_lengths(
        values=stacked_images, row_lengths=video_lengths)
    y = enc.fit_transform(video_labels).toarray()

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    return dataset


def generate_train_dataset(dataframe,
                           columns,
                           video_ids,
                           train_map_fn,
                           batch_size=32,
                           buffer_size=5000,
                           deterministic=False):
    # convert dataframe to dataset
    ds = dataframe_to_dataset(dataframe, columns, video_ids)

    # shuffle, map and batch dataset
    if deterministic:
        train_dataset = ds \
            .shuffle(buffer_size) \
            .map(train_map_fn) \
            .batch(batch_size)
    else:
        train_dataset = ds \
            .shuffle(buffer_size) \
            .map(train_map_fn,
                 num_parallel_calls=tf.data.AUTOTUNE,
                 deterministic=False) \
            .batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)

    return train_dataset


def generate_test_dataset(dataframe,
                          columns,
                          video_ids,
                          test_map_fn,
                          batch_size=32):
    # convert dataframe to dataset
    ds = dataframe_to_dataset(dataframe, columns, video_ids)

    # batch dataset
    max_element_length = dataframe \
        .groupby("video").size().max()
    bucket_boundaries = list(range(1, max_element_length))
    bucket_batch_sizes = [batch_size] * max_element_length
    ds = ds.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x)[0],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        no_padding=True)

    # map dataset
    dataset = ds \
        .map(test_map_fn,
             num_parallel_calls=tf.data.AUTOTUNE,
             deterministic=False) \
        .cache()

    return dataset


class SplitDataset():
    def __init__(self, train_dataframe, validation_dataframe, num_splits=None):
        # retrieve the joints order
        _, _, joints_order = tssi_v2()

        # define the preprocessing
        # for the test dataset
        self.test_preprocessing = tf.keras.Sequential([
            tf.keras.layers.Rescaling(scale=255.0, offset=0.0),
            PadIfLessThan(frames=INPUT_HEIGHT)
        ], name="preprocessing")

        # generate train+validation dataframe
        validation_dataframe["video"] = validation_dataframe["video"] + \
            train_dataframe["video"].max() + 1
        train_and_validation_dataframe = pd.concat(
            [train_dataframe, validation_dataframe], axis=0, ignore_index=True)

        # preprocess the train+validation dataframe
        main_dataframe = preprocess_dataframe(train_and_validation_dataframe,
                                              with_root=True,
                                              with_midhip=False)

        # obtain characteristics of the dataset
        num_total_examples = len(main_dataframe["video"].unique())
        labels = main_dataframe.groupby("video")["label"].unique().tolist()

        # generate k-fold cross validator
        skf = StratifiedKFold(num_splits, shuffle=True, random_state=RANDOM_SEED)
        splits = list(
            skf.split(np.zeros(num_total_examples), labels))
        num_train_examples = len(splits[0][0])

        # expose variables
        self.joints_order = joints_order
        self.main_dataframe = main_dataframe
        self.num_total_examples = num_total_examples
        self.labels = labels
        self.splits = splits
        self.num_train_examples = num_train_examples

        # free memory
        del train_dataframe
        del validation_dataframe
        del train_and_validation_dataframe

    def get_training_set(self, split=1, batch_size=32,
                         buffer_size=5000, deterministic=False,
                         augmentations=None):
        # obtain train indices
        split_indices = self.splits[split]
        train_indices = split_indices[0]

        # define preprocessing
        # for the train dataset
        train_preprocessing = tf.keras.Sequential([
            tf.keras.layers.Rescaling(scale=255.0, offset=0.0),
        ], name="preprocessing")

        # define length_normalization layers
        # NOTE: if applied, random speed augmentation
        # changes the length of the samples a priori
        train_length_normalization = tf.keras.Sequential([
            PadIfLessThan(frames=INPUT_HEIGHT),
            ResizeIfMoreThan(frames=INPUT_HEIGHT)
        ], name="length_normalization")

        # define the list of augmentations
        # in the default order
        if augmentations == "all":
            augmentations = augmentations_order

        if augmentations == None:
            augmentations = []

        # define the augmentation layers
        # based on the list of augmentations
        layers = [available_augmentations[aug] for aug in augmentations]
        train_augmentation = tf.keras.Sequential(layers, name="augmentation")

        # define the train map function
        @tf.function
        def train_map_fn(x, y):
            batch = tf.expand_dims(x, axis=0)
            batch = train_preprocessing(batch)
            batch = train_augmentation(batch, training=True)
            x = train_length_normalization(batch)[0]
            x = tf.ensure_shape(x, [INPUT_HEIGHT, INPUT_WIDTH, 3])
            return x, y

        dataset = generate_train_dataset(self.main_dataframe,
                                         self.joints_order,
                                         train_indices,
                                         train_map_fn,
                                         batch_size=batch_size,
                                         buffer_size=buffer_size,
                                         deterministic=deterministic)

        return dataset

    def get_testing_set(self, split=1, batch_size=32):
        # obtain train indices
        split_indices = self.splits[split]
        val_indices = split_indices[1]

        # define the preprocessing
        # for the test dataset
        test_preprocessing = tf.keras.Sequential([
            tf.keras.layers.Rescaling(scale=255.0, offset=0.0),
            PadIfLessThan(frames=INPUT_HEIGHT)
        ], name="preprocessing")

        # define the train map function
        @tf.function
        def test_map_fn(x, y):
            x = x.to_tensor()
            x = test_preprocessing(x)
            return x, y

        dataset = generate_test_dataset(self.main_dataframe,
                                        self.joints_order,
                                        val_indices,
                                        test_map_fn,
                                        batch_size=batch_size)

        return dataset
