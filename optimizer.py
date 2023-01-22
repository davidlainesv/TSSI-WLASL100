from tensorflow_addons.optimizers import TriangularCyclicalLearningRate
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.optimizers import SGD

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
