import tensorflow as tf
import wandb

class LearningRateVsLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=None, eval_each_steps=10,
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
                and (self.eval_each_steps > 0) \
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
