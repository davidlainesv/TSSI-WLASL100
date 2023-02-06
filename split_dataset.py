import tensorflow as tf
from config import INPUT_WIDTH, RANDOM_SEED
from data_augmentation import RandomFlip, RandomScale, RandomShift, RandomRotation, RandomSpeed
from preprocessing import PadIfLessThan, ResizeIfMoreThan, normalize_dataframe, preprocess_dataframe
from skeleton_graph import tssi_v2
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from preprocessing import Normalization
from sklearn.model_selection import StratifiedKFold


available_augmentations_legacy = {
    'scale': RandomScale(min_value=0.0, max_value=1.0, seed=1),
    'shift': RandomShift(min_value=0.0, max_value=1.0, seed=2),
    'flip': RandomFlip("horizontal", min_value=0.0, max_value=1.0, seed=3),
    'rotation': RandomRotation(factor=15.0, min_value=0.0, max_value=1.0, seed=4),
    'speed': RandomSpeed(frames=128, seed=5)
}

available_augmentations_from_neg1_to_1 = {
    'scale': RandomScale(min_value=-1.0, max_value=1.0, seed=1),
    'shift': RandomShift(min_value=-1.0, max_value=1.0, seed=2),
    'flip': RandomFlip("horizontal", min_value=-1.0, max_value=1.0, seed=3),
    'rotation': RandomRotation(factor=15.0, min_value=-1.0, max_value=1.0, seed=4),
    'speed': RandomSpeed(frames=128, seed=5)
}

augmentations_order_legacy = ['scale', 'shift', 'flip', 'rotation', 'speed']
augmentations_order = ['flip', 'rotation', 'speed']


def dataframe_to_dataset(dataframe, columns, encoder, filter_video_ids=[]):
    x_sorted_columns = [col + "_x" for col in columns]
    y_sorted_columns = [col + "_y" for col in columns]

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
    y = encoder.transform(video_labels).toarray()

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    return dataset


def generate_train_dataset(dataframe,
                           columns,
                           train_map_fn,
                           label_encoder,
                           video_ids=[],
                           repeat=False,
                           batch_size=32,
                           buffer_size=5000,
                           deterministic=False):
    # convert dataframe to dataset
    ds = dataframe_to_dataset(dataframe, columns, label_encoder, video_ids)

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

    if repeat:
        train_dataset = train_dataset.repeat()

    return train_dataset


def generate_test_dataset(dataframe,
                          columns,
                          test_map_fn,
                          label_encoder,
                          video_ids=[],
                          batch_size=32):
    # convert dataframe to dataset
    ds = dataframe_to_dataset(dataframe, columns, label_encoder, video_ids)

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
        skf = StratifiedKFold(num_splits, shuffle=True,
                              random_state=RANDOM_SEED)
        splits = list(
            skf.split(np.zeros(num_total_examples), labels))
        num_train_examples = len(splits[0][0])
        num_val_examples = len(splits[0][1])

        # generate label encoder
        self.label_encoder = OneHotEncoder()
        video_labels = main_dataframe.groupby(
            "video")["label"].unique().tolist()
        self.label_encoder.fit(video_labels)

        # expose variables
        self.joints_order = joints_order
        self.main_dataframe = preprocess_dataframe(main_dataframe)
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.num_total_examples = num_total_examples
        self.splits = splits

    def get_training_set(self,
                         split=1,
                         batch_size=32,
                         buffer_size=5000,
                         repeat=False,
                         deterministic=False,
                         augmentations=[],
                         input_height=128,
                         normalization=Normalization.Neg1To1):
        # obtain train indices
        split_indices = self.splits[split]
        train_indices = split_indices[0]

        # preprocess the train dataframe
        train_dataframe = normalize_dataframe(self.train_dataframe,
                                              normalization=normalization)

        # define the length_normalization layers
        train_length_normalization = tf.keras.Sequential([
            PadIfLessThan(frames=input_height),
            ResizeIfMoreThan(frames=input_height)
        ], name="length_normalization")

        # define the list of augmentations
        # in the default order
        if augmentations == "all":
            if normalization == Normalization.Neg1To1:
                augmentations = augmentations_order
                available_augmentations = available_augmentations_from_neg1_to_1
            elif normalization == Normalization.Legacy:
                augmentations = augmentations_order_legacy
                available_augmentations = available_augmentations_legacy
            else:
                raise Exception(f"Unknown normalization: {normalization}")

        # define the augmentation layers
        # based on the list of augmentations
        layers = [available_augmentations[aug] for aug in augmentations]
        train_augmentation = tf.keras.Sequential(layers, name="augmentation")

        # define the train map function
        @tf.function
        def train_map_fn(x, y):
            batch = tf.expand_dims(x, axis=0)
            batch = train_augmentation(batch, training=True)
            x = train_length_normalization(batch)[0]
            x = tf.ensure_shape(x, [input_height, INPUT_WIDTH, 3])
            return x, y

        dataset = generate_train_dataset(train_dataframe,
                                         self.joints_order,
                                         train_map_fn,
                                         self.label_encoder,
                                         video_ids=train_indices,
                                         repeat=repeat,
                                         batch_size=batch_size,
                                         buffer_size=buffer_size,
                                         deterministic=deterministic)

        return dataset

    def get_validation_set(self,
                           split=1,
                           batch_size=32,
                           min_height=128,
                           max_height=256,
                           normalization=Normalization.Neg1To1):
        # obtain train indices
        split_indices = self.splits[split]
        val_indices = split_indices[1]

        # preprocess the validation dataframe
        val_dataframe = normalize_dataframe(self.validation_dataframe,
                                            normalization=normalization)

        # define the preprocessing
        # for the test dataset
        val_preprocessing = tf.keras.Sequential([
            PadIfLessThan(frames=min_height),
            ResizeIfMoreThan(frames=max_height)
        ], name="preprocessing")

        # define the train map function
        @tf.function
        def val_map_fn(x, y):
            x = x.to_tensor()
            x = val_preprocessing(x)
            return x, y

        dataset = generate_test_dataset(val_dataframe,
                                        self.joints_order,
                                        val_map_fn,
                                        self.label_encoder,
                                        video_ids=val_indices,
                                        batch_size=batch_size)

        return dataset
