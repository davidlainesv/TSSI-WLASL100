import argparse
from config import DENSENET_INPUT_SHAPE, GENERIC_INPUT_SHAPE, LRRT_LOSS_MIN_DELTA, LRRT_STOP_FACTOR, MOBILENET_INPUT_SHAPE, NASNET_INPUT_SHAPE, RANDOM_SEED
from callbacks import LearningRateVsLossCallback
from dataset import Dataset
import numpy as np
import wandb
import tensorflow as tf
import pandas as pd
from model import build_densenet121_model, build_efficientnet_model, build_mobilenetv2_model, build_nasnetmobile_model
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
    train_dataset = dataset.get_training_set(
        batch_size=config['batch_size'],
        buffer_size=dataset.num_train_examples,
        repeat=True,
        deterministic=True,
        pipeline=config['pipeline'])

    # generate val dataset
    validation_dataset = dataset.get_validation_set(
        batch_size=config['batch_size'],
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
    elif config['backbone'] == "efficientnet":
        model = build_efficientnet_model(input_shape=GENERIC_INPUT_SHAPE,
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
        loss_min_delta=config['loss_min_delta'],
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
    wandb.agent(sweep_id, project=project, entity=entity, function=agent_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Learning rate range test.')
    parser.add_argument('--entity',
                        type=str,
                        help='Entity',
                        default='cv_inside')
    parser.add_argument('--project',
                        type=str,
                        help='Project name',
                        default='lrrt-wlasl100-tssi')
    parser.add_argument('--sweep_id',
                        type=str,
                        help='Sweep id',
                        required=True)

    args = parser.parse_args()
    print(args)
    main(args)
