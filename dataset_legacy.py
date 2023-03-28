import tensorflow as tf
from config import MAX_INPUT_HEIGHT, MIN_INPUT_HEIGHT
from data_augmentation import RandomFlip, RandomScale, RandomShift, RandomRotation, RandomSpeed
from preprocessing import Center, FillBlueWithAngle, PadIfLessThan, ResizeIfMoreThan, TranslationScaleInvariant, preprocess_dataframe
from skeleton_graph import tssi_legacy, tssi_v2, tssi_v3
from sklearn.preprocessing import OneHotEncoder
import numpy as np

AugmentationDict = {
    'speed': RandomSpeed(min_frames=48, max_frames=74, seed=5),
    'rotation': RandomRotation(factor=15.0, min_value=0.0, max_value=1.0, seed=4),
    'flip': RandomFlip("horizontal", min_value=0.0, max_value=1.0, seed=3),
    'scale': RandomScale(min_value=0.0, max_value=1.0, seed=1),
    'shift': RandomShift(min_value=0.0, max_value=1.0, seed=2)
}

NormalizationDict = {
    'invariant_frame': TranslationScaleInvariant(level="frame"),
    'invariant_joint': TranslationScaleInvariant(level="joint"),
    'center': Center(around_index=0),
    'train_resize': ResizeIfMoreThan(frames=MIN_INPUT_HEIGHT),
    'test_resize': ResizeIfMoreThan(frames=MAX_INPUT_HEIGHT),
    'pad': PadIfLessThan(frames=MIN_INPUT_HEIGHT),
    'angle': FillBlueWithAngle(x_channel=0, y_channel=1, scale_to=[0, 1]),
    'norm': tf.keras.layers.Normalization(axis=-1,
                                          mean=[0.485, 0.456, 0.406],
                                          variance=[0.052441, 0.050176, 0.050625])
}

# default_augmentation_order = ['speed', 'rotation', 'flip', 'scale', 'shift']
PipelineDict = {
    'default': {
        'augmentation': ['speed', 'flip', 'scale'],
        'train_normalization': ['train_resize', 'pad'],
        'test_normalization': ['test_resize', 'pad']
    },
    'default_speed': {
        'augmentation': ['speed'],
        'train_normalization': ['train_resize', 'pad'],
        'test_normalization': ['test_resize', 'pad']
    },
    'default_flip': {
        'augmentation': ['flip'],
        'train_normalization': ['train_resize', 'pad'],
        'test_normalization': ['test_resize', 'pad']
    },
    'default_scale': {
        'augmentation': ['scale'],
        'train_normalization': ['train_resize', 'pad'],
        'test_normalization': ['test_resize', 'pad']
    },

    'default_norm': {
        'augmentation': ['speed', 'flip', 'scale'],
        'train_normalization': ['train_resize', 'pad', 'norm'],
        'test_normalization': ['test_resize', 'pad', 'norm']
    },
    'default_norm_speed': {
        'augmentation': ['speed'],
        'train_normalization': ['train_resize', 'pad', 'norm'],
        'test_normalization': ['test_resize', 'pad', 'norm']
    },
    'default_norm_flip': {
        'augmentation': ['flip'],
        'train_normalization': ['train_resize', 'pad', 'norm'],
        'test_normalization': ['test_resize', 'pad', 'norm']
    },
    'default_norm_scale': {
        'augmentation': ['scale'],
        'train_normalization': ['train_resize', 'pad', 'norm'],
        'test_normalization': ['test_resize', 'pad', 'norm']
    },

    'invariant_frame': {
        'augmentation': ['speed', 'rotation', 'flip'],
        'train_normalization': ['invariant_frame', 'pad'],
        'test_normalization': ['test_resize', 'invariant_frame', 'pad']
    },
    'invariant_joint': {
        'augmentation': ['speed', 'rotation', 'flip'],
        'train_normalization': ['invariant_joint', 'pad'],
        'test_normalization': ['test_resize', 'invariant_joint', 'pad']
    },
    'invariant_frame_center': {
        'augmentation': ['speed', 'rotation', 'flip'],
        'train_normalization': ['invariant_frame', 'center', 'pad'],
        'test_normalization': ['test_resize', 'invariant_frame', 'center', 'pad']
    },
    'center_invariant_frame': {
        'augmentation': ['speed', 'rotation', 'flip'],
        'train_normalization': ['center', 'invariant_frame', 'pad'],
        'test_normalization': ['test_resize', 'center', 'invariant_frame', 'pad']
    },
    'default_center': {
        'augmentation': ['speed', 'rotation', 'flip', 'scale'],
        'train_normalization': ['center', 'pad'],
        'test_normalization': ['test_resize', 'center', 'pad']
    },
    'default_angle': {
        'augmentation': ['speed', 'rotation', 'flip', 'scale', 'shift'],
        'train_normalization': ['angle', 'pad'],
        'test_normalization': ['test_resize', 'angle', 'pad']
    },

    'default_angle_norm': {
        'augmentation': ['speed', 'flip', 'scale'],
        'train_normalization': ['train_resize', 'angle', 'pad', 'norm'],
        'test_normalization': ['test_resize', 'angle', 'pad', 'norm']
    },

    'default_center_norm': {
        'augmentation': ['speed', 'flip', 'scale'],
        'train_normalization': ['center', 'pad', 'norm'],
        'test_normalization': ['test_resize', 'center', 'pad', 'norm']
    },

    'ablation_speed_default_center': {
        'augmentation': ['rotation', 'flip', 'scale'],
        'train_normalization': ['train_resize', 'center', 'pad'],
        'test_normalization': ['test_resize', 'center', 'pad']
    },
    'ablation_rotation_default_center': {
        'augmentation': ['speed', 'flip', 'scale'],
        'train_normalization': ['center', 'pad'],
        'test_normalization': ['test_resize', 'center', 'pad']
    },
    'ablation_flip_default_center': {
        'augmentation': ['speed', 'rotation', 'scale'],
        'train_normalization': ['center', 'pad'],
        'test_normalization': ['test_resize', 'center', 'pad']
    },
    'ablation_scale_default_center': {
        'augmentation': ['speed', 'rotation', 'flip', 'scale'],
        'train_normalization': ['center', 'pad'],
        'test_normalization': ['test_resize', 'center', 'pad']
    },

    'ablation_speed_default_norm': {
        'augmentation': ['flip', 'scale'],
        'train_normalization': ['pad', 'norm'],
        'train_normalization': ['train_resize', 'pad', 'norm'],
        'test_normalization': ['test_resize', 'pad', 'norm']
    },
    'ablation_flip_default_norm': {
        'augmentation': ['speed', 'scale'],
        'train_normalization': ['pad', 'norm'],
        'test_normalization': ['test_resize', 'pad', 'norm']
    },
    'ablation_scale_default_norm': {
        'augmentation': ['speed', 'flip'],
        'train_normalization': ['pad', 'norm'],
        'test_normalization': ['test_resize', 'pad', 'norm']
    }
}


def dataframe_to_dataset(dataframe, columns_order, encoder):
    x_sorted_columns = [col + "_x" for col in columns_order]
    y_sorted_columns = [col + "_y" for col in columns_order]

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
                           columns_order,
                           label_encoder,
                           train_map_fn,
                           repeat=False,
                           batch_size=32,
                           buffer_size=5000,
                           deterministic=False):
    # convert dataframe to dataset
    ds = dataframe_to_dataset(
        dataframe, columns_order, label_encoder)

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
                          columns_order,
                          label_encoder,
                          test_map_fn,
                          batch_size=32):
    # convert dataframe to dataset
    ds = dataframe_to_dataset(
        dataframe, columns_order, label_encoder)

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
    # augmentation: None, str or list
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


def build_normalization_pipeline(normalization):
    # normalization: None, str or list
    if normalization == None:
        layers = []
    elif type(normalization) is str:
        layers = [NormalizationDict[normalization]]
    if type(normalization) is list:
        layers = [NormalizationDict[norm]
                  for norm in normalization]
    else:
        raise Exception("Normalization " +
                        str(normalization) + " not found")

    pipeline = tf.keras.Sequential(layers, name="normalization")
    return pipeline


class Dataset():
    def __init__(self, train_dataframe, validation_dataframe, test_dataframe=None, tssi="v2"):
        # retrieve the joints and the joints order
        if tssi == "legacy":
            graph, joints_order = tssi_legacy()
        elif tssi == "v3":
            graph, joints_order = tssi_v3()
        else:
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
        labels = train_dataframe.groupby("video")["label"].unique().tolist()
        self.label_encoder = OneHotEncoder()
        self.label_encoder.fit(labels)

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
        self.input_width = int(len(joints_order))

    def get_training_set(self,
                         batch_size=32,
                         buffer_size=5000,
                         repeat=False,
                         deterministic=False,
                         augmentation=True,
                         pipeline="default"):
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

        # define the train map function
        @tf.function
        def train_map_fn(x, y):
            batch = tf.expand_dims(x, axis=0)
            batch = augmentation_pipeline(batch, training=True)
            batch = normalization_pipeline(batch, training=True)
            x = tf.ensure_shape(
                batch[0], [MIN_INPUT_HEIGHT, self.input_width, 3])
            return x, y

        dataset = generate_train_dataset(self.train_dataframe,
                                         self.joints_order,
                                         self.label_encoder,
                                         train_map_fn,
                                         repeat=repeat,
                                         batch_size=batch_size,
                                         buffer_size=buffer_size,
                                         deterministic=deterministic)

        return dataset

    def get_validation_set(self,
                           batch_size=32,
                           pipeline="default"):
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

        dataset = generate_test_dataset(self.validation_dataframe,
                                        self.joints_order,
                                        self.label_encoder,
                                        test_map_fn,
                                        batch_size=batch_size)

        return dataset

    def get_testing_set(self,
                        batch_size=32,
                        normalization=None,
                        pipeline=None):
        if self.test_dataframe is None:
            return None

        # define normalization pipeline
        if type(pipeline) is str:
            normalization = PipelineDict[pipeline]['test_normalization']
        normalization_pipeline = build_normalization_pipeline(normalization)

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            batch_x = batch_x.to_tensor()
            batch_x = normalization_pipeline(batch_x)
            return batch_x, batch_y

        dataset = generate_test_dataset(self.test_dataframe,
                                        self.joints_order,
                                        self.label_encoder,
                                        test_map_fn,
                                        batch_size=batch_size)

        return dataset