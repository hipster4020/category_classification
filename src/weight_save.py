# -*- coding: utf-8 -*-
import hydra
import numpy as np
import pandas as pd
import ray
import tensorflow as tf
import tqdm
from tensorflow.keras import layers
from transformers import AutoTokenizer

ray.init(num_cpus=48)

physical_devices = tf.config.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


def db_load(train_dir):
    df = pd.read_csv(train_dir)

    return df


@ray.remote
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm.tqdm(sequences):
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0,
        )

    # Iterate over each positive skip-gram pair to produce training examples
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
        context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes=context_class,
            num_true=1,
            num_sampled=num_ns,
            unique=True,
            range_max=vocab_size,
            seed=seed,
            name="negative_sampling",
        )

    # Build context and label vectors (for one target word)
    negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

    context = tf.concat([context_class, negative_sampling_candidates], 0)
    label = tf.constant([1] + [0] * num_ns, dtype="int64")

    # Append each element from the training example to global lists.
    targets.append(target_word)
    contexts.append(context)
    labels.append(label)

    return targets, contexts, labels


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=1, name="w2v_embedding"
        )
        self.context_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=4 + 1
        )

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum("be,bce->bc", word_emb, context_emb)
        # dots: (batch, context)

        return dots

    def custom_loss(x_logit, y_true):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


@hydra.main(config_name="config.yml")
def main(cfg):
    df = db_load(cfg.ETC.train_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer_dir)
    e = tokenizer(
        df.content.tolist(),
        max_length=512,
    )

    feature = [
        generate_training_data.remote(
            sequences=e["input_ids"],
            window_size=10,
            num_ns=4,
            vocab_size=tokenizer.vocab_size,
            seed=42,
        )
        for i in range(48)
    ]

    targets, contexts, labels = [], [], []
    for t, c, l in ray.get(feature):
        targets += t
        contexts += c
        labels += l

    print(targets)
    print(contexts)
    print(labels)

    targets = np.array(targets)
    contexts = np.array(contexts)[:, :, 0]
    labels = np.array(labels)

    print("\n")
    print(f"targets.shape: {targets.shape}")
    print(f"contexts.shape: {contexts.shape}")
    print(f"labels.shape: {labels.shape}")

    BATCH_SIZE = 4096
    BUFFER_SIZE = 10000

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    embedding_dim = 128
    word2vec = Word2Vec(tokenizer.vocab_size, embedding_dim)
    word2vec.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    word2vec.fit(dataset, epochs=20)

    # save
    weights = word2vec.target_embedding.get_weights()[0]
    np.save(cfg.ETC.weights_dir, weights)


if __name__ == "__main__":
    main()
