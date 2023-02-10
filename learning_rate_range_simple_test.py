import argparse
from config import DENSENET_INPUT_SHAPE, LRRT_LOSS_MIN_DELTA, LRRT_STOP_FACTOR, MAX_INPUT_HEIGHT, MIN_INPUT_HEIGHT, MOBILENET_INPUT_SHAPE, NASNET_INPUT_SHAPE, RANDOM_SEED
from callbacks import LearningRateVsLossCallback
from dataset import Dataset
import numpy as np
import wandb
import tensorflow as tf
import pandas as pd
from model import build_densenet121_model, build_mobilenetv2_model, build_nasnetmobile_model
from optimizer import build_sgd_optimizer

# Load data
train_dataframe = pd.read_csv("wlasl100_skeletons_train.csv", index_col=0)
validation_dataframe = pd.read_csv("wlasl100_skeletons_val.csv", index_col=0)
dataset = Dataset(train_dataframe, validation_dataframe)
del train_dataframe, validation_dataframe


def run_experiment(config=None, log_to_wandb=True, verbose=0):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # check if config was provided
    if config is None:
        print("Not config provided.")
        return
    print("[INFO] Configuration:", config, "\n")

    # generate train dataset
    augmentations = "all" if config['augmentation'] else []
    train_dataset = dataset.get_training_set(
        input_height=MIN_INPUT_HEIGHT,
        batch_size=config['batch_size'],
        buffer_size=5000,
        repeat=True,
        deterministic=True,
        pipeline=config['pipeline'])

    # generate val dataset
    validation_dataset = dataset.get_validation_set(
        batch_size=config['batch_size'],
        min_height=MIN_INPUT_HEIGHT,
        max_height=MAX_INPUT_HEIGHT,
        pipeline=config['pipeline'])

    print("[INFO] Dataset Total examples:", dataset.num_total_examples)
    print("[INFO] Dataset Training examples:", dataset.num_train_examples)
    print("[INFO] Dataset Validation examples:", dataset.num_val_examples)

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
    elif config['backbone'] == 'nasnet':
        model = build_nasnetmobile_model(input_shape=NASNET_INPUT_SHAPE,
                                         dropout=config['dropout'],
                                         optimizer=optimizer,
                                         pretraining=config['pretraining'])
    else:
        raise Exception("Unknown model name")

    # setup callback
    lrc = LearningRateVsLossCallback(
        validation_data=validation_dataset,
        log_each_steps=config['log_each_steps'],
        stop_factor=LRRT_STOP_FACTOR,
        stop_patience=config['stop_patience'],
        loss_min_delta=LRRT_LOSS_MIN_DELTA,
        log_to_wandb=log_to_wandb)

    # train model
    model.fit(train_dataset,
              epochs=1,
              steps_per_epoch=int(config['step_size']),
              verbose=verbose,
              callbacks=[lrc])

    # get the logs of the callback
    logs = lrc.get_logs()

    return logs


def agent_fn(config=None):
    wandb.init(config=config, reinit=True)

    maximal_learning_rate = wandb.config.maximal_learning_rate
    initial_learning_rate = wandb.config.initial_learning_rate
    learning_rate_delta = wandb.config.learning_rate_delta
    batch_size = wandb.config.batch_size

    learning_rate_distance = maximal_learning_rate - initial_learning_rate
    step_size = learning_rate_distance / learning_rate_delta
    log_each_steps = np.ceil(dataset.num_train_examples / batch_size)

    update = {"step_size": step_size, "log_each_steps": log_each_steps}
    wandb.config.update(update)

    _ = run_experiment(config=wandb.config, log_to_wandb=True, verbose=0)

    wandb.finish()


def main(args):
    entity = args.entity
    project = args.project
    sweep_id = args.sweep_id
    backbone = args.backbone
    pretraining = args.pretraining
    augmentation = args.augmentation
    lr_min = args.lr_min
    lr_max = args.lr_max
    lr_delta = args.lr_delta
    repetitions = args.repetitions
    normalization = args.normalization
    stop_patience = args.stop_patience

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
                'initial_learning_rate': {'value': lr_min},
                'maximal_learning_rate': {'value': lr_max},
                'learning_rate_delta': {'value': lr_delta},
                'dropout': {'values': [0.1, 0.3, 0.5]},
                'weight_decay': {'values': [1e-4, 1e-5, 1e-6, 1e-7]},
                'batch_size': {'values': [32, 64, 128]},
                'repetitions': {'values': list(range(1, repetitions + 1))},
                'normalization': {'value': normalization},
                'momentum': {'value': 0.9},
                'nesterov': {'value': True},
                'stop_patience': {'value': stop_patience}
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
                        help='Project name', default='lrrt-wlasl100-tssi')
    parser.add_argument('--sweep_id', type=str, help='Sweep id')
    parser.add_argument('--backbone', type=str,
                        help='Backbone method: \'densenet\', \'mobilenet\'')
    parser.add_argument('--pretraining', type=bool, help='Add pretraining')
    parser.add_argument('--augmentation', type=bool, help='Add augmentation')
    parser.add_argument('--lr_min', type=float, help='Minimum learning rate')
    parser.add_argument('--lr_max', type=float, help='Maximal learning rate')
    parser.add_argument('--lr_delta', type=float,
                        help='Learning rate increment at every step')
    parser.add_argument('--repetitions', type=int, help='Repetitions')
    parser.add_argument('--normalization', type=str,
                        help='Normalization method (\'neg1_to_1\', \'zero_to_1\'')
    parser.add_argument('--stop_patience', type=int,
                        help='Stop patience', default=10)
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
        if args.lr_delta is None:
            raise Exception("Please provide lr_delta")
        if args.repetitions is None:
            raise Exception("Please provide repetitions")
        if args.normalization is None:
            raise Exception("Please provide normalization")

    print(args)

    main(args)
