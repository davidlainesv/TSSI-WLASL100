import tensorflow as tf
from config import MIN_INPUT_HEIGHT, RANDOM_SEED
from dataset import PipelineDict, build_augmentation_pipeline, build_normalization_pipeline, generate_test_dataset, generate_train_dataset
from preprocessing import filter_dataframe_by_video_ids, preprocess_dataframe
from skeleton_graph import tssi_legacy, tssi_v2, tssi_v3
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class SplitDataset():
    def __init__(self, train_dataframe, validation_dataframe, num_splits=None, tssi="v2"):
        # retrieve the joints and the joints order
        if tssi == "legacy":
            graph, joints_order = tssi_legacy()
        elif tssi == "v3":
            graph, joints_order = tssi_v3()
        else:
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
        labels = main_dataframe.groupby("video")["label"].unique().tolist()

        # generate k-fold cross validator
        skf = StratifiedKFold(num_splits, shuffle=True,
                              random_state=RANDOM_SEED)
        splits = list(skf.split(np.zeros(num_total_examples), labels))

        # obtain characteristics of the dataset
        num_train_examples = len(splits[0][0])
        num_val_examples = len(splits[0][1])

        # generate label encoder
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
        self.input_width = int(len(joints_order))

    def get_training_set(self,
                         split=1,
                         batch_size=32,
                         buffer_size=5000,
                         repeat=False,
                         deterministic=False,
                         augmentation=True,
                         pipeline=None):
        # obtain train indices
        split_indices = self.splits[split-1]
        train_indices = split_indices[0]

        # define pipeline
        if type(pipeline) is str:
            augmentation_layers = PipelineDict[pipeline]['augmentation'] \
                if augmentation else []
            normalization_layers = PipelineDict[pipeline]['train_normalization']
        else:
            raise Exception("Pipeline not provided")
        augmentation_pipeline = build_augmentation_pipeline(
            augmentation_layers)
        normalization_pipeline = build_normalization_pipeline(
            normalization_layers)

        # filter main dataframe using train_indices
        train_dataframe = filter_dataframe_by_video_ids(
            self.main_dataframe, train_indices)

        # define the train map function
        @tf.function
        def train_map_fn(x, y):
            batch = tf.expand_dims(x, axis=0)
            batch = augmentation_pipeline(batch, training=True)
            batch = normalization_pipeline(batch, training=True)
            x = tf.ensure_shape(batch[0], [MIN_INPUT_HEIGHT, self.input_width, 3])
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
                           pipeline=None):
        # obtain train indices
        split_indices = self.splits[split-1]
        val_indices = split_indices[1]

        # filter main dataframe using val_indices
        val_dataframe = filter_dataframe_by_video_ids(
            self.main_dataframe, val_indices)

        # define normalization pipeline
        if type(pipeline) is str:
            normalization_layers = PipelineDict[pipeline]['test_normalization']
        else:
            raise Exception("Pipeline not provided")
        normalization_pipeline = build_normalization_pipeline(
            normalization_layers)

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            batch_x = batch_x.to_tensor()
            batch_x = normalization_pipeline(batch_x)
            return batch_x, batch_y

        dataset = generate_test_dataset(val_dataframe,
                                        self.joints_order,
                                        self.label_encoder,
                                        test_map_fn,
                                        batch_size=batch_size)

        return dataset
