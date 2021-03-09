import os
import time
import logging

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

from srcs.args.ArgParser import ArgParser
from srcs.tools.ImageHandler import ImageHandler


class GanHandler(ArgParser, ImageHandler):
    EPOCHS = 50
    NOISE_DIM = 100
    BATCH_SIZE = 256
    BUFFER_SIZE = 60000
    NUM_EX_TO_GENERATE = 16
    checkpoint_dir = "./training_checkpoints"

    @staticmethod
    def _options_args(gan_parser):
        gan_parser.add_argument(
            "-d",
            "--directory",
            type=str,
            help="Model directory",
            default=f"mnist_gan_dir/",
            dest="directory",
        )
        gan_parser.add_argument(
            "-f",
            "--filename",
            type=str,
            help=f"Image filename [if a filename is provided, the file is saved]",
            default=None,
            dest="filename",
        )

    def _add_subparser_args(self, parser):
        super()._add_subparser_args(parser)
        subparser = parser.add_subparsers(help="MNIST_GAN")
        gan_parser = subparser.add_parser(name="GAN")
        self._options_args(gan_parser)

    @staticmethod
    def _generate_discriminator_model():
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

        return model

    @staticmethod
    def _generate_generator_model():
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

        return model

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_output):
        logging.warning(f"fake output {fake_output}")
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def _train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self._generator_loss(fake_output)
            logging.warning(f"GENERATOR LOSS: {gen_loss}")
            disc_loss = self._discriminator_loss(real_output, fake_output)
            logging.warning(f"DISCRIMINATOR LOSS: {disc_loss}")

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

    def _init_dataset(self):
        (tmp_train_imgs, _), (_, _) = tf.keras.datasets.mnist.load_data()
        tmp_train_imgs = tmp_train_imgs.reshape(tmp_train_imgs.shape[0], 28, 28, 1).astype(
            "float32"
        )
        # preprocess images
        tmp_train_imgs = (tmp_train_imgs - 127.5) / 127.5
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices(tmp_train_imgs)
            .shuffle(self.BUFFER_SIZE)
            .batch(self.BATCH_SIZE)
        )

    def _init_optimizer(self):
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def _init_checkpoint(self):
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
        )

    def _init_options(self):
        self.m_path = os.path.join(self.args.directory, str(time.time()).replace(".", "_"))
        logging.info(f"Using dir path: {self.m_path}")
        os.makedirs(self.m_path)

    def __init__(self):
        super().__init__()
        self._init_dataset()
        self._init_options()
        self.generator = self._generate_generator_model()
        self.discriminator = self._generate_discriminator_model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._init_optimizer()
        self._init_checkpoint()

    def generate_and_save_images(self, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.generator(test_input, training=False)
        # fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
            plt.axis("off")
        plt.savefig(f"{self.m_path}/image_at_epoch_{epoch:04d}.png")

    def train(self, epochs: int = EPOCHS):
        seed = tf.random.normal([self.NUM_EX_TO_GENERATE, self.NOISE_DIM])
        for epoch in range(epochs):
            start = time.time()

            for image_batch in self.train_dataset:
                self._train_step(image_batch)

            self.generate_and_save_images(epoch + 1, seed)

            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            logging.info(f"Time epoch {epoch + 1}: {time.time() - start:.1f}s")

    def generate(self, noise=None, show: bool = False):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        start = time.time()
        if noise is None:
            noise = tf.random.normal([1, 100])
        generated_img = self.generator(noise, training=False)[0, :, :, 0]
        logging.info(f"Generation time: {time.time() - start:.1f}s")
        if self.args.filename is not None:
            self.save_gif(self.m_path)
        if show:
            plt.imshow(generated_img, cmap="gray")
            plt.show()
        return generated_img
