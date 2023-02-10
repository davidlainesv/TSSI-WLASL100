import tensorflow as tf
from config import INPUT_WIDTH, MAX_INPUT_HEIGHT, MIN_INPUT_HEIGHT
from data_augmentation import RandomFlip, RandomScale, RandomShift, RandomRotation, RandomSpeed
from preprocessing import Center, PadIfLessThan, ResizeIfMoreThan, TranslationScaleInvariant, preprocess_dataframe
from skeleton_graph import tssi_v2
from sklearn.preprocessing import OneHotEncoder
import numpy as np

AugmentationDict = {
    'speed': RandomSpeed(min_frames=60, max_frames=MIN_INPUT_HEIGHT, seed=5),
    'rotation': RandomRotation(factor=15.0, min_value=0.0, max_value=1.0, seed=4),
    'flip': RandomFlip("horizontal", min_value=0.0, max_value=1.0, seed=3),
    'scale': RandomScale(min_value=0.0, max_value=1.0, seed=1),
    'shift': RandomShift(min_value=0.0, max_value=1.0, seed=2),
    'all': [
        RandomSpeed(min_frames=60, max_frames=MIN_INPUT_HEIGHT, seed=5),
        RandomRotation(factor=15.0, min_value=0.0, max_value=1.0, seed=4),
        RandomFlip("horizontal", min_value=0.0, max_value=1.0, seed=3),
        RandomScale(min_value=0.0, max_value=1.0, seed=1),
        RandomShift(min_value=0.0, max_value=1.0, seed=2)
    ]
}

SpaceNormalizationDict = {
    'invariant_frame': TranslationScaleInvariant(level="frame"),
    'invariant_joint': TranslationScaleInvariant(level="joint"),
    'center': Center(around_index=0)
}

NormalizationDict = {
    'invariant_frame': TranslationScaleInvariant(level="frame"),
    'invariant_joint': TranslationScaleInvariant(level="joint"),
    'center': Center(around_index=0),
    'train_resize': ResizeIfMoreThan(frames=MIN_INPUT_HEIGHT),
    'test_resize': ResizeIfMoreThan(frames=MAX_INPUT_HEIGHT),
    'pad': PadIfLessThan(frames=MIN_INPUT_HEIGHT)
}

PipelineDict = {
    'default': {
        'augmentation': ['speed', 'rotation', 'flip', 'scale', 'shift'],
        'space_normalization': [],
        'train_normalization': [],
        'test_normalization': ['test_resize', 'pad']
    },
    'invariant_frame': {
        'augmentation': ['speed', 'rotation', 'flip'],
        'space_normalization': ['invariant_frame'],
        'train_normalization': ['invariant_frame', 'pad'],
        'test_normalization': ['test_resize', 'pad']
    },
    'invariant_joint': {
        'augmentation': ['speed', 'rotation', 'flip'],
        'space_normalization': ['invariant_joint'],
        'train_normalization': ['invariant_joint', 'pad'],
        'test_normalization': ['test_resize', 'pad']
    },
    'invariant_frame_center': {
        'augmentation': ['speed', 'rotation', 'flip'],
        'space_normalization': ['invariant_frame', 'center'],
        'train_normalization': ['invariant_frame', 'center', 'pad'],
        'test_normalization': ['test_resize', 'pad']
    },
    'center_invariant_frame': {
        'augmentation': ['speed', 'rotation', 'flip'],
        'space_normalization': ['center', 'invariant_frame'],
        'train_normalization': ['center', 'invariant_frame', 'pad'],
        'test_normalization': ['test_resize', 'pad']
    },
    'default_center': {
        'augmentation': ['speed', 'rotation', 'flip', 'scale'],
        'space_normalization': ['center'],
        'train_normalization': ['center', 'pad'],
        'test_normalization': ['test_resize', 'pad']
    }
}


def dataframe_to_dataset(dataframe, ordered_columns, encoder, filter_video_ids=[]):
    x_sorted_columns = [col + "_x" for col in ordered_columns]
    y_sorted_columns = [col + "_y" for col in ordered_columns]

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
                           ordered_columns,
                           train_map_fn,
                           label_encoder,
                           video_ids=[],
                           repeat=False,
                           batch_size=32,
                           buffer_size=5000,
                           deterministic=False):
    # convert dataframe to dataset
    ds = dataframe_to_dataset(
        dataframe, ordered_columns, label_encoder, video_ids)

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
                          ordered_columns,
                          test_map_fn,
                          label_encoder,
                          video_ids=[],
                          batch_size=32):
    # convert dataframe to dataset
    ds = dataframe_to_dataset(
        dataframe, ordered_columns, label_encoder, video_ids)

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


def build_augmentation_pipeline(augmentation):
    # augmentation: 'all', None or list
    if augmentation == None:
        layers = []
    elif type(augmentation) is str:
        layers = [AugmentationDict[augmentation]]
    elif type(augmentation) is list:
        layers = [AugmentationDict[aug] for aug in augmentation]
    else:
        raise Exception("Augmentation " + str(augmentation) + " not found")
    pipeline = tf.keras.Sequential(layers, name="augmentation")
    return pipeline


def build_normalization_pipeline(space_normalization, min_height, max_height):
    # space normalization: None or list
    # it normalizes between 0 and 1 based on min and max values
    # it may subtract root joint
    if space_normalization == None:
        layers = []
    elif type(space_normalization) is str:
        layers = [SpaceNormalizationDict[space_normalization]]
    if type(space_normalization) is list:
        layers = [SpaceNormalizationDict[norm]
                  for norm in space_normalization]
    else:
        raise Exception("Normalization " +
                        str(space_normalization) + " not found")

    # time normalization does not have effect
    # when speed augmentation is in augmentations
    # in other case, it may change min and max values
    layers = layers + [ResizeIfMoreThan(frames=max_height)]

    # padding with 0's must be applied after space normalization
    # to not affect the minimum and maximum values
    # and represent no movement with 0's
    layers = layers + [
        PadIfLessThan(frames=min_height)
    ]

    pipeline = tf.keras.Sequential(layers, name="normalization")
    return pipeline


class Dataset():
    def __init__(self, train_dataframe, validation_dataframe, test_dataframe=None):
        # retrieve the joints and the joints order
        graph, joints_order = tssi_v2()
        columns = [joint + "_x" for joint in graph.nodes]
        columns += [joint + "_y" for joint in graph.nodes]

        # obtain characteristics of the dataset
        num_train_examples = len(train_dataframe["video"].unique())
        num_val_examples = len(validation_dataframe["video"].unique())
        num_total_examples = num_train_examples + num_val_examples
        if test_dataframe is not None:
            num_test_examples = len(test_dataframe["video"].unique())
        else:
            num_test_examples = 0

        # generate label encoder
        self.label_encoder = OneHotEncoder()
        video_labels = train_dataframe.groupby(
            "video")["label"].unique().tolist()
        self.label_encoder.fit(video_labels)

        # expose variables
        self.joints_order = joints_order
        self.train_dataframe = preprocess_dataframe(
            train_dataframe, select_columns=columns)
        self.validation_dataframe = preprocess_dataframe(
            validation_dataframe, select_columns=columns)
        self.test_dataframe = preprocess_dataframe(
            test_dataframe, select_columns=columns) if test_dataframe is not None else None
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.num_test_examples = num_test_examples
        self.num_total_examples = num_total_examples

    def get_training_set(self,
                         input_height=128,
                         batch_size=32,
                         buffer_size=5000,
                         repeat=False,
                         deterministic=False,
                         augmentation=[],
                         space_normalization=[],
                         pipeline=None):
        # define augmentation+normalization pipeline
        if type(pipeline) is str:
            augmentation = PipelineDict[pipeline]['augmentation']
            space_normalization = PipelineDict[pipeline]['space_normalization']
        augmentation_pipeline = build_augmentation_pipeline(augmentation)
        normalization_pipeline = build_normalization_pipeline(
            space_normalization, input_height, input_height)

        # define the train map function
        @tf.function
        def train_map_fn(x, y):
            batch = tf.expand_dims(x, axis=0)
            batch = augmentation_pipeline(batch, training=True)
            batch = normalization_pipeline(batch, training=True)
            x = tf.ensure_shape(batch[0], [input_height, INPUT_WIDTH, 3])
            return x, y

        dataset = generate_train_dataset(self.train_dataframe,
                                         self.joints_order,
                                         train_map_fn,
                                         self.label_encoder,
                                         video_ids=[],
                                         repeat=repeat,
                                         batch_size=batch_size,
                                         buffer_size=buffer_size,
                                         deterministic=deterministic)

        return dataset

    def get_validation_set(self,
                           batch_size=32,
                           min_height=128,
                           max_height=256,
                           space_normalization=None,
                           pipeline=None):
        # define normalization pipeline
        if type(pipeline) is str:
            space_normalization = PipelineDict[pipeline]['space_normalization']
        normalization_pipeline = build_normalization_pipeline(
            space_normalization, min_height, max_height)

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            batch_x = batch_x.to_tensor()
            batch_x = normalization_pipeline(batch_x)
            return batch_x, batch_y

        dataset = generate_test_dataset(self.validation_dataframe,
                                        self.joints_order,
                                        test_map_fn,
                                        self.label_encoder,
                                        video_ids=[],
                                        batch_size=batch_size)

        return dataset

    def get_testing_set(self,
                        batch_size=32,
                        min_height=128,
                        max_height=256,
                        space_normalization=None,
                        pipeline=None):
        if self.test_dataframe is None:
            return None

        # define normalization pipeline
        if type(pipeline) is str:
            space_normalization = PipelineDict[pipeline]['space_normalization']
        normalization_pipeline = build_normalization_pipeline(
            space_normalization, min_height, max_height)

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            batch_x = batch_x.to_tensor()
            batch_x = normalization_pipeline(batch_x)
            return batch_x, batch_y

        dataset = generate_test_dataset(self.test_dataframe,
                                        self.joints_order,
                                        test_map_fn,
                                        self.label_encoder,
                                        video_ids=[],
                                        batch_size=batch_size)

        return dataset
