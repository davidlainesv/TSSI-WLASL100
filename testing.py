import argparse
from config import DENSENET_INPUT_SHAPE, GENERIC_INPUT_SHAPE, MOBILENET_INPUT_SHAPE, NASNET_INPUT_SHAPE, RANDOM_SEED
from dataset import Dataset
import numpy as np
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint
import tensorflow as tf
import pandas as pd
from model import build_densenet121_model, build_efficientnet_model, build_mobilenetv2_model, build_nasnetmobile_model
from optimizer import build_sgd_optimizer

# Load data
train_dataframe = pd.read_csv("wlasl100_skeletons_train.csv", index_col=0)
validation_dataframe = pd.read_csv("wlasl100_skeletons_val.csv", index_col=0)
test_dataframe = pd.read_csv("wlasl100_skeletons_test.csv", index_col=0)


validation_dataframe["video"] = validation_dataframe["video"] + \
    train_dataframe["video"].max() + 1
train_and_validation_dataframe = pd.concat(
    [train_dataframe, validation_dataframe], axis=0, ignore_index=True)

dataset = Dataset(train_and_validation_dataframe, test_dataframe)
del train_dataframe, validation_dataframe, test_dataframe


def run_experiment(config=None, log_to_wandb=True, verbose=0):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # check if config was provided
    if config is None:
        print("Not config provided.")
        return
    print("[INFO] Configuration:", config, "\n")

    # generate train dataset
    train_dataset = dataset.get_training_set(
        batch_size=config['batch_size'],
        buffer_size=dataset.num_train_examples,
        repeat=False,
        deterministic=True,
        pipeline=config['pipeline'])

    # generate val dataset
    validation_dataset = dataset.get_validation_set(
        batch_size=config['batch_size'],
        pipeline=config['pipeline'])

    print("[INFO] Dataset Total examples:", dataset.num_total_examples)
    print("[INFO] Dataset Training examples:", dataset.num_train_examples)
    print("[INFO] Dataset Training examples:", dataset.num_val_examples)

    # setup optimizer
    optimizer = build_sgd_optimizer(initial_learning_rate=config['initial_learning_rate'],
                                    maximal_learning_rate=config['maximal_learning_rate'],
                                    momentum=config['momentum'],
                                    nesterov=config['nesterov'],
                                    step_size=config['step_size'],
                                    weight_decay=config['weight_decay'])

    # setup model
    if config['backbone'] == "densenet":
        model = build_densenet121_model(input_shape=DENSENET_INPUT_SHAPE,
                                        dropout=config['dropout'],
                                        optimizer=optimizer,
                                        pretraining=config['pretraining'])
    elif config['backbone'] == "mobilenet":
        model = build_mobilenetv2_model(input_shape=MOBILENET_INPUT_SHAPE,
                                        dropout=config['dropout'],
                                        optimizer=optimizer,
                                        pretraining=config['pretraining'])
    elif config['backbone'] == "nasnet":
        model = build_nasnetmobile_model(input_shape=NASNET_INPUT_SHAPE,
                                         dropout=config['dropout'],
                                         optimizer=optimizer,
                                         pretraining=config['pretraining'])
    elif config['backbone'] == "efficientnet":
        model = build_efficientnet_model(input_shape=GENERIC_INPUT_SHAPE,
                                         dropout=config['dropout'],
                                         optimizer=optimizer,
                                         pretraining=config['pretraining'])
    else:
        raise Exception("Unknown model name")

    # setup callbacks
    callbacks = []
    if log_to_wandb:
        wandb_logger = WandbMetricsLogger()
        wandb_model_checkpoint = WandbModelCheckpoint(
            save_weights_only=True,
            save_freq=config['save_freq'],
            verbose=0
        )
        callbacks.append(wandb_logger)
        callbacks.append(wandb_model_checkpoint)

    # train model
    model.fit(train_dataset,
              epochs=config['num_epochs'],
              verbose=verbose,
              validation_data=validation_dataset,
              callbacks=callbacks)

    # get the logs of the model
    return model.history


def agent_fn(config, project, entity, verbose=0):
    wandb.init(entity=entity, project=project, config=config, reinit=True)
    _ = run_experiment(config=wandb.config, log_to_wandb=True, verbose=verbose)
    wandb.finish()


def main(args):
    entity = args.entity
    project = args.project
    lr_min = args.lr_min
    lr_max = args.lr_max
    backbone = args.backbone
    augmentation = args.augmentation
    pretraining = args.pretraining
    dropout = args.dropout
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    pipeline = args.pipeline
    save_freq = args.save_freq

    steps_per_epoch = np.ceil(dataset.num_train_examples / batch_size)

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
        'augmentation': augmentation,
        'batch_size': batch_size,
        'pipeline': pipeline,

        'save_freq': steps_per_epoch * save_freq
    }

    agent_fn(config=config, project=project, entity=entity, verbose=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traning and testing.')
    parser.add_argument('--entity', type=str,
                        help='Entity', default='davidlainesv')
    parser.add_argument('--project', type=str,
                        help='Project name', default='testing')
    parser.add_argument('--backbone', type=str,
                        help='Backbone method: \'densenet\', \'mobilenet\'',
                        default='densenet')
    parser.add_argument('--pretraining', type=bool,
                        help='Add pretraining', default=True)
    parser.add_argument('--augmentation', type=bool,
                        help='Add augmentation', default=False)
    parser.add_argument('--lr_min', type=float,
                        help='Minimum learning rate', default=0.0001)
    parser.add_argument('--lr_max', type=float,
                        help='Minimum learning rate', default=0.001)
    parser.add_argument('--dropout', type=float,
                        help='Minimum learning rate', default=0.3)
    parser.add_argument('--weight_decay', type=float,
                        help='Minimum learning rate', default=1e-7)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size of training and testing', default=32)
    parser.add_argument('--num_epochs', type=int,
                        help='Number of epochs', default=100)
    parser.add_argument('--pipeline', type=str,
                        help='Pipeline', default="default")
    parser.add_argument('--save_freq', type=int,
                        help='Save weights at epoch', default=100)
    args = parser.parse_args()

    print(args)

    main(args)
