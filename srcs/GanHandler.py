import os
import time

import tensorflow as tf
from IPython import display
from tensorflow.keras import layers

from srcs.ImageHandler import ImageHandler


class GanHandler(ImageHandler):
    EPOCHS = 50
    NOISE_DIM = 100
    BATCH_SIZE = 256
    BUFFER_SIZE = 60000
    NUM_EX_TO_GENERATE = 16
    checkpoint_dir = "./training_checkpoints"

    def _generate_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(
            layers.Conv2D(
                64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
            )
        )
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        self.discriminator = model

    def _generate_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(
            layers.Conv2DTranspose(
                128, (5, 5), strides=(1, 1), padding="same", use_bias=False
            )
        )
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(
            layers.Conv2DTranspose(
                64, (5, 5), strides=(2, 2), padding="same", use_bias=False
            )
        )
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(
            layers.Conv2DTranspose(
                1,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                activation="tanh",
            )
        )
        assert model.output_shape == (None, 28, 28, 1)
        self.generator = model

    def _init_dataset(self):
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
            "float32"
        )
        self.train_images = (train_images - 127.5) / 127.5
        self.dataset = (
            tf.data.Dataset.from_tensor_slices(train_images)
            .shuffle(self.BUFFER_SIZE)
            .batch(self.BATCH_SIZE)
        )

    def _init_checkpoint(self):
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
        )

    def _init_optimizer(self):
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def _train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

    def __init__(self):
        self.noise = tf.random.normal([1, 100])
        self._init_dataset()
        self._generate_generator_model()
        self._generate_discriminator_model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._init_optimizer()
        self._init_checkpoint()

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def train(self, epochs: int = EPOCHS):
        seed = tf.random.normal([self.NUM_EX_TO_GENERATE, self.NOISE_DIM])
        for epoch in range(epochs):
            start = time.time()
            for image_batch in self.dataset:
                self._train_step(image_batch)
            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator, epoch + 1, seed)
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print(f"Time for epoch {epoch + 1} is {time.time() - start} sec")

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(self.generator, epochs, seed)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def generate(self, noise=None):
        if noise is None:
            noise = tf.random.normal([1, 100])
        return self.generator(noise, training=False)
