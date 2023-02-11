import tensorflow as tf
from config import INPUT_WIDTH, MIN_INPUT_HEIGHT, RANDOM_SEED
from dataset import PipelineDict, build_augmentation_pipeline, build_normalization_pipeline, generate_test_dataset, generate_train_dataset
from preprocessing import filter_dataframe_by_video_ids, preprocess_dataframe
from skeleton_graph import tssi_v2
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class SplitDataset():
    def __init__(self, train_dataframe, validation_dataframe, num_splits=None):
        # retrieve the joints order
        graph, joints_order = tssi_v2()
        columns = [joint + "_x" for joint in graph.nodes]
        columns += [joint + "_y" for joint in graph.nodes]

        # generate train+validation dataframe
        validation_dataframe["video"] = validation_dataframe["video"] + \
            train_dataframe["video"].max() + 1
        main_dataframe = pd.concat([train_dataframe, validation_dataframe],
                                   axis=0, ignore_index=True)

        # obtain characteristics of the dataset
        num_total_examples = len(main_dataframe["video"].unique())

        # generate k-fold cross validator
        skf = StratifiedKFold(num_splits, shuffle=True,
                              random_state=RANDOM_SEED)
        splits = list(skf.split(np.zeros(num_total_examples), labels))

        # obtain characteristics of the dataset
        num_train_examples = len(splits[0][0])
        num_val_examples = len(splits[0][1])

        # generate label encoder
        labels = main_dataframe.groupby("video")["label"].unique().tolist()
        self.label_encoder = OneHotEncoder()
        self.label_encoder.fit(labels)

        # expose variables
        self.joints_order = joints_order
        self.main_dataframe = preprocess_dataframe(
            main_dataframe, select_columns=columns)
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
                         augmentation=[],
                         normalization=[],
                         pipeline=None):
        # obtain train indices
        split_indices = self.splits[split-1]
        train_indices = split_indices[0]

        # define pipeline
        if type(pipeline) is str:
            augmentation = PipelineDict[pipeline]['augmentation']
            normalization = PipelineDict[pipeline]['train_normalization']
        augmentation_pipeline = build_augmentation_pipeline(augmentation)
        normalization_pipeline = build_normalization_pipeline(normalization)

        # filter main dataframe using train_indices
        train_dataframe = filter_dataframe_by_video_ids(
            self.main_dataframe, train_indices)

        # define the train map function
        @tf.function
        def train_map_fn(x, y):
            batch = tf.expand_dims(x, axis=0)
            batch = augmentation_pipeline(batch, training=True)
            batch = normalization_pipeline(batch, training=True)
            x = tf.ensure_shape(batch[0], [MIN_INPUT_HEIGHT, INPUT_WIDTH, 3])
            return x, y

        dataset = generate_train_dataset(train_dataframe,
                                         self.joints_order,
                                         self.label_encoder,
                                         train_map_fn,
                                         repeat=repeat,
                                         batch_size=batch_size,
                                         buffer_size=buffer_size,
                                         deterministic=deterministic)

        return dataset

    def get_validation_set(self,
                           split=1,
                           batch_size=32,
                           normalization=[],
                           pipeline=None):
        # obtain train indices
        split_indices = self.splits[split-1]
        val_indices = split_indices[1]

        # filter main dataframe using val_indices
        val_dataframe = filter_dataframe_by_video_ids(
            self.main_dataframe, val_indices)

        # define normalization pipeline
        if type(pipeline) is str:
            normalization = PipelineDict[pipeline]['test_normalization']
        normalization_pipeline = build_normalization_pipeline(
            normalization)

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            batch_x = batch_x.to_tensor()
            batch_x = normalization_pipeline(batch_x)
            return batch_x, batch_y

        dataset = generate_test_dataset(val_dataframe,
                                        self.joints_order,
                                        test_map_fn,
                                        self.label_encoder,
                                        batch_size=batch_size)

        return dataset
