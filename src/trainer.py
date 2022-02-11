import os
from math import ceil
from shutil import rmtree

import tensorflow as tf
import transformers as tr
from tqdm import tqdm
from transformers.optimization_tf import create_optimizer

import wandb

wandb.init(project="em_category")

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tr.logging.set_verbosity(tr.logging.ERROR)


class TrainArgument:
    def __init__(self, **kwargs):
        # training parameters
        self.strategy = self.get_strategy(kwargs.get("use_gpu", True))
        self.train_batch_size = kwargs.get("train_batch_size", 4)
        self.train_global_batch_size = (
            self.train_batch_size * self.strategy.num_replicas_in_sync
        )
        self.eval_batch_size = kwargs.get("eval_batch_size", 4)
        self.eval_global_batch_size = (
            self.eval_batch_size * self.strategy.num_replicas_in_sync
        )
        self.epochs = kwargs.get("epochs", 1)
        self.eval_epoch = kwargs.get("eval_epoch", self.epochs)
        self.signature = self.set_signature(kwargs.get("signature"))

        # checkpoint
        self.checkpoint_dir = kwargs.get("checkpoint_dir")
        self.save_epoch = kwargs.get("save_epoch", 1)
        self.save_total_limit = kwargs.get("save_total_limit", int(1e9))

        # logging
        self.logging_dir = kwargs.get("logging_dir")
        self.logging_steps = kwargs.get("logging_steps", 100)
        self.logging_print = kwargs.get("logging_print", False)

        # optimizer
        self.learning_rate = kwargs.get("learning_rate", 5e-05)
        self.warmup_steps = kwargs.get("warmup_steps", 0)
        self.adam_beta1 = kwargs.get("adam_beta1", 0.9)
        self.adam_beta2 = kwargs.get("adam_beta2", 0.98)
        self.adam_epsilon = kwargs.get("adam_epsilon", 1e-9)

    def get_strategy(self, use_gpu):
        gpus = tf.config.list_physical_devices("GPU")

        if use_gpu:
            if len(gpus) == 0:
                strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            elif len(gpus) == 1:
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            elif len(gpus) > 1:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.ReductionToOneDevice()
                )
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

        return strategy

    def set_signature(self, signature=None):
        return (
            {
                k: tf.TensorSpec(shape=eval(v["shape"]), dtype=v["type"])
                for k, v in signature.items()
            }
            if signature is not None
            else None
        )


class Trainer:
    def __init__(
        self,
        model,
        args,
        train_dataset,
        loss_function,
        eval_dataset=None,
        data_collator=None,
        optimizers=[None, None],
        metrics=None,
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.loss_function = loss_function

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.do_eval = self.eval_dataset is not None

        self.optimizer, self.lr_scheduler = optimizers
        self.set_tensorboard(self.args.logging_dir)

        self.set_metrics(metrics)

    def __del__(self):
        wandb.finish()

    def set_metrics(self, metrics=None):
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.now_epoch = 0

        metrics = [metrics] if hasattr(metrics, "__call__") else metrics
        for i in range(len(metrics)):
            if not hasattr(metrics[i], "__name__"):
                metrics[i].__name__ = metrics[i].__class__.__name__

        if isinstance(metrics, list) or isinstance(metrics, tuple):
            self.metrics_func = metrics
            self.metrics = [
                tf.keras.metrics.Mean(name=m.__name__) for m in self.metrics_func
            ]
        else:
            self.metrics_func = None
            self.train_metrics = None
            self.eval_metrics = None

    def set_checkpoint(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            self.checkpoint = False
            return

        self.checkpoint = True
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

    def save_checkpoint(self, epoch):
        assert self.checkpoint
        ckpt_list = os.listdir(self.args.checkpoint_dir)
        save_dir = os.path.join(self.args.checkpoint_dir, f"epoch_{epoch}")
        if self.args.logging_print:
            print("saved at " + save_dir)

        if self.args.save_total_limit <= len(ckpt_list):
            first_one = os.path.join(self.args.checkpoint_dir, sorted(ckpt_list)[0])
            rmtree(first_one)

        self.model.save(save_dir)

    def set_tensorboard(self, logging_dir=None):
        if logging_dir is None:
            self.logging = False
            return

        self.logging = True

        self.logger = tf.summary.create_file_writer(logging_dir)

    def set_optimizer(self, optimizer=None, lr_scheduler=None):
        num_training_steps = (
            ceil(len(self.train_dataset) / self.args.train_global_batch_size)
            * self.args.epochs
        )

        if optimizer is None:
            self.optimizer, self.lr_scheduler = create_optimizer(
                self.args.learning_rate,
                num_training_steps,
                self.args.warmup_steps,
                adam_beta1=self.args.adam_beta1,
                adam_beta2=self.args.adam_beta2,
                adam_epsilon=self.args.adam_epsilon,
            )

        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

    def get_dataset(self, dataset, signature=None, batch_size=None):
        batch_size = batch_size if batch_size else self.args.train_global_batch_size

        # dataset.set_format(type="tensorflow")
        # features = {
        #     k: tf.cast(
        #         dataset[k].to_tensor(shape=signature[k].shape), dtype=signature[k].dtype
        #     )
        #     for k in dataset.column_names
        # }
        # dataset_tf = tf.data.Dataset.from_tensor_slices(features)

        def _gen():
            for data in dataset:
                yield data

        dataset_tf = tf.data.Dataset.from_generator(_gen, output_signature=signature)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF
        )

        dataset_tf = dataset_tf.batch(batch_size)
        if self.data_collator is not None:
            dataset_tf = dataset_tf.map(self.data_collator).prefetch(
                tf.data.experimental.AUTOTUNE
            )
        dataset_tf.with_options(options)

        return self.args.strategy.experimental_distribute_dataset(dataset_tf)

    def log(self, log_dict, step):
        self.logger.flush()
        wandb.log(log_dict)
        with self.logger.as_default():
            for name, value in log_dict.items():
                tf.summary.scalar(name, value, step=step)

    @tf.function
    def step(self, data, training=False):
        pred = self.model(data, training=training)
        # pred = self.model(**data, training=training)
        loss = self.loss_function(data["labels"], pred)
        if self.metrics_func is not None:
            metrics = [m(data["labels"], pred) for m in self.metrics_func]

        if training:
            gradients = tf.gradients(loss, self.model.trainable_variables)
            gradients = [
                g if g is not None else tf.zeros_like(v)
                for g, v in zip(gradients, self.model.trainable_variables)
            ]
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

        self.loss(loss)
        if self.metrics_func is not None:
            for i, m in enumerate(metrics):
                self.metrics[i](m)

    @tf.function
    def distributed_step(self, data, training=False):
        self.args.strategy.run(self.step, args=(data, training))

    def train(self, dataset=None, signature=None):
        if dataset is None:
            signature = self.args.signature
            dataset = self.train_dataset
        else:
            assert (
                signature is not None
            ), "data signature must be required to loop custom datasets."

        batch_size = self.args.train_global_batch_size

        self.set_checkpoint(self.args.checkpoint_dir)
        # TODO: checkpoint 넣어주면 로드할 수 있게 하기
        with self.args.strategy.scope():
            if self.optimizer is None:
                self.set_optimizer(self.optimizer, self.lr_scheduler)

            step_per_epoch = ceil(len(dataset) / batch_size)

            pbar = tqdm(total=step_per_epoch * self.args.epochs)
            dataset = self.get_dataset(dataset, signature, batch_size=batch_size)

            for epoch in range(self.args.epochs):
                self.now_epoch = epoch
                self.loss.reset_states()
                if self.metrics_func is not None:
                    for m in self.metrics:
                        m.reset_states()

                for step, data in enumerate(dataset):
                    global_step = step_per_epoch * epoch + step

                    self.distributed_step(data, training=True)

                    if self.logging and global_step % self.args.logging_steps == 0:
                        log_dict = dict()
                        tag = "/train"

                        log_dict["epoch" + tag] = global_step / step_per_epoch
                        if self.lr_scheduler is not None:
                            lr = self.lr_scheduler(global_step).numpy()
                            log_dict["lr" + tag] = lr

                        log_dict["loss" + tag] = self.loss.result()

                        if self.metrics_func is not None:
                            for m in self.metrics:
                                log_dict[m.name + tag] = m.result()

                        self.log(log_dict, global_step)

                        if self.args.logging_print:
                            print(
                                "train step {}: {}".format(
                                    global_step,
                                    ", ".join(
                                        [f"{k}: {v: .4f}" for k, v in log_dict.items()]
                                    ),
                                )
                            )

                    pbar.update(1)

                if self.checkpoint and (epoch + 1) % self.args.save_epoch == 0:
                    str_len = len(str(self.args.epochs))
                    epoch_str = str(epoch)
                    epoch_str = "0" * (str_len - len(epoch_str)) + epoch_str
                    self.save_checkpoint(epoch_str)

                if self.do_eval and self.now_epoch % self.args.eval_epoch == 0:
                    self.eval(view_progress=False)

    def eval(self, dataset=None, signature=None, view_progress=True):
        if dataset is None:
            assert self.eval_dataset is not None, "eval dataset is not exist."
            dataset = self.eval_dataset
            signature = self.args.signature
        else:
            assert (
                signature is not None
            ), "data signature must be required to loop custom datasets."
        batch_size = self.args.eval_global_batch_size

        with self.args.strategy.scope():
            step_per_epoch = ceil(len(dataset) / batch_size)

            if view_progress:
                pbar = tqdm(total=step_per_epoch)
            dataset = self.get_dataset(dataset, signature, batch_size=batch_size)

            self.loss.reset_states()
            if self.metrics_func is not None:
                for m in self.metrics:
                    m.reset_states()

            for data in dataset:
                self.distributed_step(data, training=False)

                if view_progress:
                    pbar.update(1)

        tag = "/eval"
        log_dict = {"loss" + tag: self.loss.result()}
        if self.metrics_func is not None:
            for m in self.metrics:
                log_dict[m.name + tag] = m.result()

        if self.logging:
            self.log(log_dict, self.now_epoch)

        return log_dict
