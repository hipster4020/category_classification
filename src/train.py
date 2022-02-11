import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra
from transformers import AutoTokenizer

from dataloader import default_collator, load
from metrics import CustomF1Score, accuracy, loss
from models.MainModels import EncoderModel
from trainer import TrainArgument, Trainer


@hydra.main(config_name="config.yml")
def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer_dir)
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    args = TrainArgument(**cfg.TRAINARGS)
    with args.strategy.scope():
        model = EncoderModel(vocab_size=tokenizer.vocab_size, **cfg.MODEL)
        model(model._get_sample_data())

        weights = np.load(cfg.ETC.weights_dir)
        model.embedding.embedding.set_weights([weights])
        model.embedding.trainable = False

        metrics = [
            accuracy,
            CustomF1Score(cfg.MODEL.num_classes, average="micro"),
        ]

    trainer = Trainer(
        model,
        args,
        train_dataset,
        loss,
        eval_dataset=eval_dataset,
        data_collator=default_collator,
        metrics=metrics,
    )

    trainer.train()

    model.save(cfg.ETC.output_dir)


if __name__ == "__main__":
    main()
