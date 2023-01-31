import tensorflow as tf
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet121_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input


def build_densenet121_model(input_shape=[None, 181, 3], dropout=0,
                            optimizer=None, pretraining=True):
    # setup model
    weights = 'imagenet' if pretraining else None
    inputs = Input(shape=input_shape)
    x = DenseNet121(input_shape=input_shape, weights=weights,
                    include_top=False, pooling='avg')(inputs)
    x = Dropout(dropout)(x)
    predictions = Dense(100, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=metrics)

    return model


def build_mobilenetv2_model(input_shape=[None, 181, 3], dropout=0,
                            optimizer=None, pretraining=True):
    # setup model
    weights = "imagenet" if pretraining else None
    inputs = Input(shape=input_shape)
    inputs = mobilenetv2_preprocess_input(inputs)
    x = MobileNetV2(input_shape=input_shape, weights=weights,
                    include_top=False, pooling="avg")(inputs)
    x = Dropout(dropout)(x)
    predictions = Dense(100, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=metrics)

    return model
