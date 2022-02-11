import tensorflow as tf

from models.UtilLayers import FFNN, MultiHeadAttention


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rate = rate

        # multi-head attention
        self.mha_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.mha_dropout = tf.keras.layers.Dropout(rate)

        # ffnn
        self.ffnn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffnn = FFNN(d_model, d_model * 4, "relu", rate)
        self.ffnn_dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False):
        # extended_mask
        extended_mask = mask[:, tf.newaxis, tf.newaxis, :] if mask is not None else None

        # multi-head attention
        mha_output = self.mha_norm(x, training=training)
        mha_output, _ = self.mha(mha_output, mha_output, mha_output, extended_mask)
        mha_output = self.mha_dropout(mha_output, training=training)
        mha_output += x

        # ffnn
        ffnn_output = self.ffnn_norm(mha_output, training=training)
        ffnn_output = self.ffnn(ffnn_output, training=training)
        ffnn_output = self.ffnn_dropout(ffnn_output, training=training)
        ffnn_output += mha_output

        return ffnn_output

    def get_config(self):
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "rate": self.rate,
        }


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rate = rate

        # cross multi-head attention with encoder
        self.mha1_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha1_dropout = tf.keras.layers.Dropout(rate)

        # cross multi-head attention with encoder
        self.mha2_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.mha2_dropout = tf.keras.layers.Dropout(rate)

        # ffnn
        self.ffnn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffnn = FFNN(d_model, d_model * 2, "relu", rate)
        self.ffnn_dropout = tf.keras.layers.Dropout(rate)

    def call(
        self,
        x,
        enc_output,
        look_ahead_mask,
        padding_mask,
        training=False,
        **kwargs,
    ):
        # extended_mask
        extended_look_ahead_mask = look_ahead_mask

        extended_padding_mask = (
            padding_mask[:, tf.newaxis, tf.newaxis, :]
            if padding_mask is not None
            else None
        )

        # cross multi-head attention with encoder
        mha1_output = self.mha1_norm(x, training=training)
        mha1_output, _ = self.mha1(
            mha1_output, mha1_output, mha1_output, extended_look_ahead_mask
        )
        mha1_output = self.mha1_dropout(mha1_output, training=training)
        mha1_output += x

        # cross multi-head attention with encoder
        mha2_output = self.mha1_norm(mha1_output, training=training)
        mha2_output, _ = self.mha1(
            mha2_output, enc_output, enc_output, extended_padding_mask
        )
        mha2_output = self.mha1_dropout(mha2_output, training=training)
        mha2_output += mha1_output

        # ffnn
        ffnn_output = self.ffnn_norm(mha2_output, training=training)
        ffnn_output = self.ffnn(ffnn_output, training=training)
        ffnn_output = self.ffnn_dropout(ffnn_output, training=training)
        ffnn_output += mha2_output

        return ffnn_output

    def get_config(self):
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "rate": self.rate,
        }
