import argparse
from config import INPUT_SHAPE, LEARNING_RATE_STEP, NUM_SPLITS, RANDOM_SEED
from callbacks import LearningRateVsLossCallback
from dataset import generate_dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import wandb
import tensorflow as tf
import pandas as pd
from model import build_densenet121_model, build_mobilenetv2_model

from optimizer import build_sgd_optimizer


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
              epochs=1,
              steps_per_epoch=config["optimizer"]["step_size"],
              verbose=verbose,
              callbacks=[lrc])

    # get the logs of the callback
    logs = lrc.get_logs()

    return logs


def agent_fn(config=None):
    run = wandb.init(config=config, reinit=True)
    step_size = int((wandb.config.initial_learning_rate -
                 wandb.config.maximal_learning_rate) / LEARNING_RATE_STEP)
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
            'step_size': step_size
        },
        'training': {
            'train_batch_size': wandb.config.batch_size,
            'test_batch_size': wandb.config.batch_size,
            'augmentation':  wandb.config.augmentation,
            'eval_each_steps': 1,
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
                'nesterov': {'value': True}
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
