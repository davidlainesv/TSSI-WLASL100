# -*- coding: utf-8 -*-
"""LRRT-0_1to1_0-wo_pretraining-wth_augmentation.ipynb
"""

import argparse
from config import INPUT_SHAPE, NUM_SPLITS, RANDOM_SEED
from dataset import generate_dataset, augmentations_order
from sklearn.model_selection import StratifiedKFold
import numpy as np
import wandb
from wandb.keras import WandbCallback
from model import build_densenet121_model, build_mobilenetv2_model
from optimizer import build_sgd_optimizer
import tensorflow as tf
import pandas as pd


# """ Download data"""
# !wget "https://storage.googleapis.com/cloud-ai-platform-f3305919-42dc-47f1-82cf-4f1a3202db74/wlasl100_skeletons_train.csv" -nc
# !wget "https://storage.googleapis.com/cloud-ai-platform-f3305919-42dc-47f1-82cf-4f1a3202db74/wlasl100_skeletons_val.csv" -nc
# !wget "https://storage.googleapis.com/cloud-ai-platform-f3305919-42dc-47f1-82cf-4f1a3202db74/wlasl100_skeletons_test.csv" -nc

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


def run_experiment(config=None, log_to_wandb=True, verbose=0):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # check if config was provided
    if config is None:
        print("Not config provided.")
        return
    print("[INFO] Configuration:", config, "\n")

    # select split (1...NUM_SPLITS)
    print("Training on split {}".format(config["split"]))
    split_indices = splits[config["split"] - 1]
    train_indices, val_indices = split_indices

    # generate dataset
    if config["ablation"] == "none":
        augmentations = "all"
    else:
        augmentations = augmentations_order.copy()
        augmentations.remove(config["ablation"])
    train_dataset = generate_dataset(
        train_and_validation_dataframe,
        video_ids=train_indices,
        training=True,
        batch_size=config['train_batch_size'],
        buffer_size=5000,
        deterministic=True,
        augmentations=augmentations)

    # generate val dataset
    validation_dataset = generate_dataset(
        train_and_validation_dataframe,
        video_ids=val_indices,
        training=False,
        batch_size=config['test_batch_size'])

    # setup optimizer
    optimizer = build_sgd_optimizer(initial_learning_rate=config['initial_learning_rate'],
                                    maximal_learning_rate=config['maximal_learning_rate'],
                                    step_size=config['step_size'], momentum=config['momentum'],
                                    nesterov=config['nesterov'], weight_decay=config['weight_decay'])

    # setup model
    if config['backbone'] == "densenet":
        model = build_densenet121_model(input_shape=INPUT_SHAPE,
                                        dropout=config['dropout'],
                                        optimizer=optimizer,
                                        pretraining=config['pretraining'])
    elif config['backbone'] == "mobilenet":
        model = build_mobilenetv2_model(input_shape=INPUT_SHAPE,
                                        dropout=config['dropout'],
                                        optimizer=optimizer,
                                        pretraining=config['pretraining'])

    # setup callback
    callbacks = []
    if log_to_wandb:
        wandb_callback = WandbCallback(
            monitor="val_top_1",
            mode="max",
            save_model=False
        )
        callbacks = [wandb_callback]

    # train model
    model.fit(train_dataset,
              validation_data=validation_dataset,
              epochs=config['num_epochs'],
              verbose=verbose,
              callbacks=callbacks)


def main(args):
    available_ablations = ['scale', 'shift',
                           'flip', 'rotation', 'speed', 'none']

    entity = args.entity
    project = args.project
    lr_min = args.lr_min
    lr_max = args.lr_max
    backbone = args.backbone
    pretraining = args.pretraining
    dropout = args.dropout
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    ablations = [abl for abl in args.ablations.split()
                 if abl in available_ablations]

    steps_per_epoch = np.ceil(num_train_examples / batch_size)

    for ablation in ablations:
        for split in list(range(1, NUM_SPLITS + 1)):
            config = {
                'backbone': backbone,
                'pretraining': pretraining,
                'dropout': dropout,

                'initial_learning_rate': lr_min,
                'maximal_learning_rate': lr_max,
                'momentum': 0.9,
                'nesterov': True,
                'weight_decay': weight_decay,
                'step_size': int(num_epochs / 2) * steps_per_epoch,

                'num_epochs': num_epochs,
                'train_batch_size': batch_size,
                'test_batch_size': batch_size,
                'ablation': ablation,
                'split': split
            }

            run = wandb.init(reinit=True, entity=entity,
                             project=project, config=config)
            run_experiment(config=config, log_to_wandb=True, verbose=0)
            run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning rate range test.')
    parser.add_argument('--entity', type=str,
                        help='Entity', default='cv_inside')
    parser.add_argument('--project', type=str,
                        help='Project name', required=True)
    parser.add_argument('--ablations', type=str,
                        help='Ablations (performed individually): "scale", "shift", "flip", "rotation", "speed", "none"',
                        required=True)
    parser.add_argument('--backbone', type=str,
                        help='Backbone method: \'densenet\', \'mobilenet\'',
                        required=True)
    parser.add_argument('--pretraining', type=bool, help='Add pretraining',
                        required=True)
    parser.add_argument('--lr_min', type=float, help='Minimum learning rate',
                        required=True)
    parser.add_argument('--lr_max', type=float, help='Maximum learning rate',
                        required=True)
    parser.add_argument('--weight_decay', type=float, help='Weight decay',
                        required=True)
    parser.add_argument('--dropout', type=float, help='Dropout',
                        required=True)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size (train & test)', required=True)
    parser.add_argument('--num_epochs', type=int,
                        help='Number of epochs', required=True)
    args = parser.parse_args()

    main(args)
