import numpy as np
import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, pe, rate=0.1):
        super(PositionalEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pe = pe
        self.rate = rate
        self.positional_encoding = self.get_positional_encoding(
            self.pe, self.embedding_size
        )
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.dropout = tf.keras.layers.Dropout(self.rate)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def get_positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        return x

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "pe": self.pe,
            "rate": self.rate,
        }


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += tf.cast((1 - mask), tf.float32) * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        logit, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        logit = tf.transpose(logit, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(logit, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights

    def get_config(self):
        return {
            "num_,heads": self.num_heads,
            "d_model": self.d_model,
        }


class FFNN(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, activation, rate=0.1):
        super(FFNN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.rate = rate

        self.dense1 = tf.keras.layers.Dense(d_ff, activation=activation)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        output = self.dense2(x)
        return output

    def get_config(self):
        return {
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "activation": self.activation,
            "rate": self.rate,
        }
