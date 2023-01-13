# -*- coding: utf-8 -*-
"""LRRT-0_1to1_0-wo_pretraining-wth_augmentation.ipynb
"""

import argparse
from dataset import generate_dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import wandb
from tensorflow_addons.optimizers import TriangularCyclicalLearningRate
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet121_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
import tensorflow as tf
import pandas as pd


# """ Download data"""
# !wget "https://storage.googleapis.com/cloud-ai-platform-f3305919-42dc-47f1-82cf-4f1a3202db74/wlasl100_skeletons_train.csv" -nc
# !wget "https://storage.googleapis.com/cloud-ai-platform-f3305919-42dc-47f1-82cf-4f1a3202db74/wlasl100_skeletons_val.csv" -nc
# !wget "https://storage.googleapis.com/cloud-ai-platform-f3305919-42dc-47f1-82cf-4f1a3202db74/wlasl100_skeletons_test.csv" -nc

NUM_SPLITS = 5
INPUT_SHAPE = [None, 181, 3]
RANDOM_SEED = 0

# Load data
train_dataframe = pd.read_csv("wlasl100_skeletons_train.csv", index_col=0)
validation_dataframe = pd.read_csv("wlasl100_skeletons_val.csv", index_col=0)
validation_dataframe["video"] = validation_dataframe["video"] + \
    train_dataframe["video"].max() + 1
train_and_validation_dataframe = pd.concat(
    [train_dataframe, validation_dataframe], axis=0, ignore_index=True)

# Split data
skf = StratifiedKFold(NUM_SPLITS)
num_total_examples = len(train_and_validation_dataframe["video"].unique())
labels = train_and_validation_dataframe.groupby(
    "video")["label"].unique().tolist()
splits = list(skf.split(np.zeros(num_total_examples), labels))
num_train_examples = len(splits[0][0])


class LearningRateVsLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=None, eval_each_steps=0,
                 stop_factor=4, stop_patience=10, loss_min_delta=0.1,
                 log_to_wandb=False, add_to_log={}):
        super(LearningRateVsLossCallback, self).__init__()
        self.lrs = []
        self.stop_factor = stop_factor or 1e9
        self.stop_patience = stop_patience or 1e9
        self.loss_min_delta = loss_min_delta or 0

        self.batch_num = 0
        self.train_best_loss = 1e9

        self.eval_num = 0
        self.eval_each_steps = eval_each_steps
        self.validation_data = validation_data
        self.val_best_loss = 1e9
        self.patience_num = 0

        self.logs = []
        self.add_to_log = add_to_log
        self.log_to_wandb = log_to_wandb

    def on_train_batch_begin(self, batch, logs=None):
        # grab the learning rate used for this batch
        # and add it to the learning rate history
        lr = self.model.optimizer._decayed_lr('float32')
        self.lrs.append(lr)

    def on_train_batch_end(self, batch, logs=None):
        # grab the learning rate used for this batch
        lr = self.lrs[-1]

        # grab the loss at the end of this batch
        loss = logs["loss"]

        # increment the total number of batches processed
        self.batch_num += 1

        # check to see if the best val loss should be updated
        if loss < self.train_best_loss:
            self.train_best_loss = loss

        # initialize the log of this batch
        log = {
            **self.add_to_log,
            **{
                'train_lr': lr,
                'train_loss': loss,
                'train_best_loss': self.train_best_loss
            }
        }

        if (self.validation_data is not None) \
                and (self.eval_each_steps >= 0) \
                and (self.batch_num % self.eval_each_steps == 0):
            # grab the loss from the evaluation
            scores = self.model.evaluate(self.validation_data,
                                         return_dict=True,
                                         verbose=0)
            val_loss = scores['loss']

            # increment the total number of evaluations
            self.eval_num += 1

            # compute the maximum loss stopping factor value
            stop_loss = self.stop_factor * self.val_best_loss

            # check to see whether the val loss has grown too large
            if self.eval_num > 1 and val_loss >= stop_loss:
                print(f"Stopped with val_loss {val_loss}, " +
                      f"stop_loss {stop_loss} and " +
                      f"val_best_loss {self.val_best_loss}")

                # stop training
                self.model.stop_training = True

            # check to see if the best val loss should be updated
            # if not, increment the patience number
            if val_loss <= (self.val_best_loss - self.loss_min_delta):
                self.val_best_loss = val_loss
                self.patience_num = 0
            else:
                self.patience_num += 1

            # check if patience has been exhausted
            if self.patience_num == self.stop_patience:
                print(f"Stopped with val_loss {val_loss}, " +
                      f"patience_num {self.patience_num} and " +
                      f"val_best_loss {self.val_best_loss}")

                # stop training
                self.model.stop_training = True

            # update the log of this batch
            log = {
                **log,
                **{
                    'val_lr': lr,
                    'val_loss': val_loss,
                    'val_best_loss': self.val_best_loss
                }
            }

        # append the log to the log history
        self.logs.append(log)

        # log to wandb
        if self.log_to_wandb:
            wandb.log(log)

    def get_logs(self):
        return self.logs


def build_sgd_optimizer(initial_learning_rate=0.001,
                        maximal_learning_rate=0.01,
                        step_size=50, momentum=0.0,
                        nesterov=False, weight_decay=1e-7):
    # setup schedule
    learning_rate_schedule = TriangularCyclicalLearningRate(
        initial_learning_rate=initial_learning_rate,
        maximal_learning_rate=maximal_learning_rate,
        step_size=step_size)

    # setup the optimizer
    if weight_decay:
        initial_weight_decay = weight_decay
        maximal_weight_decay = weight_decay * \
            (maximal_learning_rate / initial_learning_rate)
        weight_decay_schedule = TriangularCyclicalLearningRate(
            initial_learning_rate=initial_weight_decay,
            maximal_learning_rate=maximal_weight_decay,
            step_size=step_size)

        optimizer = SGDW(learning_rate=learning_rate_schedule,
                         weight_decay=weight_decay_schedule,
                         momentum=momentum, nesterov=nesterov)
    else:
        optimizer = SGD(learning_rate=learning_rate_schedule,
                        momentum=momentum, nesterov=nesterov)
    return optimizer


def build_densenet121_model(input_shape=[None, 181, 3], dropout=0,
                            optimizer=None, pretraining=True):
    # setup model
    weights = 'imagenet' if pretraining else None
    inputs = Input(shape=input_shape)
    inputs = densenet121_preprocess_input(inputs)
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


def run_experiment(config=None, log_to_wandb=True, verbose=0):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # check if config was provided
    if config is None:
        print("Not config provided.")
        return
    print("[INFO] Configuration:", config, "\n")

    # select split (1...NUM_SPLITS)
    print("Training on split {}".format(config["training"]["split"]))
    split_indices = splits[config["training"]["split"] - 1]
    train_indices, val_indices = split_indices

    # generate dataset
    augmentations = "all" if config["training"]["augmentation"] else None
    train_dataset = generate_dataset(
        train_and_validation_dataframe,
        video_ids=train_indices,
        training=True,
        batch_size=config["training"]['train_batch_size'],
        buffer_size=5000,
        deterministic=True,
        augmentations=augmentations)

    # generate val dataset
    validation_dataset = generate_dataset(
        train_and_validation_dataframe,
        video_ids=val_indices,
        training=False,
        batch_size=config["training"]['test_batch_size'])

    # setup optimizer
    optimizer = build_sgd_optimizer(initial_learning_rate=config['optimizer']['initial_learning_rate'],
                                    maximal_learning_rate=config['optimizer']['maximal_learning_rate'],
                                    momentum=config['optimizer']['momentum'],
                                    nesterov=config['optimizer']['nesterov'],
                                    step_size=config['optimizer']['step_size'],
                                    weight_decay=config['optimizer']['weight_decay'])

    # setup model
    if config['model']['backbone'] == "densenet":
        model = build_densenet121_model(input_shape=INPUT_SHAPE,
                                        dropout=config['model']['dropout'],
                                        optimizer=optimizer,
                                        pretraining=config['model']['pretraining'])
    elif config['model']['backbone'] == "mobilenet":
        model = build_mobilenetv2_model(input_shape=INPUT_SHAPE,
                                        dropout=config['model']['dropout'],
                                        optimizer=optimizer,
                                        pretraining=config['model']['pretraining'])

    # setup callback
    eval_each_steps = config["training"]['eval_each_steps']
    lrc = LearningRateVsLossCallback(
        validation_data=validation_dataset,
        eval_each_steps=eval_each_steps,
        stop_patience=10, loss_min_delta=0.1,
        log_to_wandb=log_to_wandb
    )

    # train model
    model.fit(train_dataset,
              epochs=config['training']['num_epochs'],
              verbose=verbose,
              callbacks=[lrc])

    # get the logs of the callback
    logs = lrc.get_logs()

    return logs


def agent_fn(config=None):
    run = wandb.init(config=config, reinit=True)
    steps_per_epoch = np.ceil(num_train_examples / wandb.config.batch_size)
    config = {
        'model': {
            'backbone': wandb.config.backbone,
            'pretraining': wandb.config.pretraining,
            'dropout': wandb.config.dropout
        },
        'optimizer': {
            'initial_learning_rate': wandb.config.initial_learning_rate,
            'maximal_learning_rate': wandb.config.maximal_learning_rate,
            'momentum': wandb.config.momentum,
            'nesterov': wandb.config.nesterov,
            'weight_decay': wandb.config.weight_decay,
            'step_size': wandb.config.num_epochs * steps_per_epoch
        },
        'training': {
            'num_epochs': wandb.config.num_epochs,
            'train_batch_size': wandb.config.batch_size,
            'test_batch_size': wandb.config.batch_size,
            'augmentation':  wandb.config.augmentation,
            'eval_each_steps': steps_per_epoch,
            'split': wandb.config.split
        }
    }

    _ = run_experiment(config=config, log_to_wandb=True, verbose=0)
    run.finish()


def main(args):
    entity = args.entity
    project = args.project
    sweep_id = args.sweep_id
    lr_min = args.lr_min
    lr_max = args.lr_max
    backbone = args.backbone
    augmentation = args.augmentation
    pretraining = args.pretraining

    if sweep_id is None:
        sweep_configuration = {
            'method': 'grid',
            'name': 'sweep',
            'metric': {'goal': 'minimize', 'name': 'val_best_loss'},
            'parameters':
            {
                'backbone': {'value': backbone},
                'augmentation': {'value': augmentation},
                'pretraining': {'value': pretraining},
                'split': {'values': list(range(1, NUM_SPLITS + 1))},
                'initial_learning_rate': {'value': lr_min},
                'maximal_learning_rate': {'value': lr_max},
                'batch_size': {'values': [32, 64, 128]},
                'weight_decay': {'values': [1e-4, 1e-5, 1e-6, 1e-7]},
                'dropout': {'values': [0.1, 0.3, 0.5]},
                'momentum': {'value': 0.9},
                'nesterov': {'value': True},
                'num_epochs': {'value': 50},
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration,
                               project=project, entity=entity)

    wandb.agent(sweep_id, project=project, entity=entity, function=agent_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning rate range test.')
    parser.add_argument('--entity', type=str,
                        help='Entity', default='cv_inside')
    parser.add_argument('--project', type=str,
                        help='Project name', required=True)
    parser.add_argument('--sweep_id', type=str, help='Sweep id')
    parser.add_argument('--backbone', type=str,
                        help='Backbone method: \'densenet\', \'mobilenet\'')
    parser.add_argument('--pretraining', type=bool, help='Add pretraining')
    parser.add_argument('--augmentation', type=bool, help='Add augmentation')
    parser.add_argument('--lr_min', type=float, help='Minimum learning rate')
    parser.add_argument('--lr_max', type=float, help='Minimum learning rate')
    args = parser.parse_args()

    if args.sweep_id is None:
        if args.backbone is None:
            raise Exception("Please provide backbone")
        if args.pretraining is None:
            raise Exception("Please provide pretraining")
        if args.augmentation is None:
            raise Exception("Please provide augmentation")
        if args.lr_min is None:
            raise Exception("Please provide lr_min")
        if args.lr_max is None:
            raise Exception("Please provide lr_max")
        print(args.entity, args.project, args.backbone,
              args.augmentation, args.lr_min, args.lr_max)

    print(args.entity, args.project, args.sweep_id)

    main(args)
