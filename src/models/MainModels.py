import json
import os

import tensorflow as tf

from models.TransformerLayers import EncoderLayer
from models.UtilLayers import PositionalEmbedding


class EncoderModel(tf.keras.Model):
    def __init__(
        self,
        embedding_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int,
        num_classes: int,
        seq_len: int,
        pe: int = 1024,
        rate: float = 0.1,
    ):
        super(EncoderModel, self).__init__()

        self.embedding_size = embedding_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.pe = pe
        self.rate = rate

        self.embedding = PositionalEmbedding(vocab_size, embedding_size, pe)
        if embedding_size != d_model:
            self.embedding_intermediate = tf.keras.layers.Dense(d_model)

        self.encoders = [
            EncoderLayer(d_model, num_heads, rate) for _ in range(self.num_layers)
        ]
        self.intermediate_layers = [
            tf.keras.layers.Dense(self.d_model, activation="gelu")
            for _ in range(self.num_layers)
        ]
        self.dropout = [
            tf.keras.layers.Dropout(rate=self.rate) for _ in range(self.num_layers)
        ]
        self.pooling_embedding = tf.keras.layers.Embedding(1, self.d_model)
        self.output_layer = tf.keras.layers.Dense(
            self.num_classes, activation="sigmoid"
        )

    def dot_attention(self, q, k, v, mask=None):
        logits = tf.matmul(q, k, transpose_b=True)

        if mask is not None:
            logits += tf.cast((1 - mask[:, tf.newaxis, :]), tf.float32) * -1e9

        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output

    def call(
        self,
        inputs,
        # input_ids,
        # attention_mask=None,
        training=False,
        **kwargs,
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # forward
        encoder_output = self.embedding(
            input_ids
        )  # shape [batch_size, sequance_length]
        if self.embedding_size != self.d_model:
            encoder_output = self.embedding_intermediate(encoder_output)

        for i in range(self.num_layers):

            encoder_output = self.encoders[i](
                encoder_output, attention_mask, training=training
            )
            encoder_output = self.intermediate_layers[i](encoder_output)
            encoder_output = self.dropout[i](encoder_output, training=training)
        # shape [batch_size, sequance_length, d_model]

        # add pooling layer
        pooling_code = self.pooling_embedding(  # code_embedding
            tf.range(1, dtype=tf.int32)
        )  # shape: (batch_size, 1, d_model)
        encoder_output = self.dot_attention(
            pooling_code, encoder_output, encoder_output, mask=attention_mask
        )  # shape: (batch_size, 1, d_model). query의 shape를 따라감.
        encoder_output = tf.squeeze(
            encoder_output, [1]
        )  # == tf.keras.Layers.Flatten()(encoder_output)

        output = self.output_layer(
            encoder_output
        )  # shape [batch_size, sequance_length, num_classes]
        return output

    def get_config(self):
        return {
            "embedding_size": self.embedding_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "seq_len": self.seq_len,
            "pe": self.pe,
            "rate": self.rate,
        }

    def _get_sample_data(self):
        sample_data = {
            "input_ids": tf.random.uniform((1, 8), 0, self.vocab_size, dtype=tf.int64),
        }
        return sample_data

    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.get_config(), f)

        self(self._get_sample_data())
        self.save_weights(os.path.join(save_dir, "model_weights.h5"))

        return os.listdir(save_dir)

    @classmethod
    def load(cls, save_dir):
        with open(os.path.join(save_dir, "config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        model(model._get_sample_data())
        model.load_weights(os.path.join(save_dir, "model_weights.h5"))

        return model
