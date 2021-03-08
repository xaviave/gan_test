import os
import time
import logging
import PIL.Image

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from srcs.args.ArgParser import ArgParser
from srcs.tools.ImageHandler import ImageHandler


class DeepDream(ArgParser, ImageHandler):
    """
    https://www.tensorflow.org/tutorials/generative/deepdream
    """

    layers: list
    octaves: list
    step_size: float
    octave_scale: float
    steps_per_octave: int

    def _model_args(self, dd_parser):
        dd_parser.add_argument(
            "-e",
            "--epochs",
            type=int,
            help=f"Number of epoch in each octaves",
            default=100,
            dest="epochs",
        )
        dd_parser.add_argument(
            "-s",
            "--step_size",
            type=float,
            help=f"Gradient Descent step size",
            default=1e-2,
            dest="step_size",
        )
        dd_parser.add_argument(
            "-o",
            "--octaves",
            type=int,
            nargs="+",
            help=f"List of octaves iterations",
            default=range(-2, 3),
            dest="octaves",
        )
        dd_parser.add_argument(
            "-os",
            "--octave_scale",
            type=float,
            help=f"Octave scaling",
            default=1.3,
            dest="octave_scale",
        )
        dd_parser.add_argument(
            "-ls",
            "--layers",
            type=str,
            nargs="+",
            help=f"DeepDream model is created based InceptionV3 model's layers\nList of Layers in model: {' - '.join([l.name for l in self.m_.layers])}",
            default=["mixed3", "mixed5"],
            dest="layers",
        )

    @staticmethod
    def _options_args(dd_parser):
        dd_parser.add_argument(
            "-ss",
            "--show_step",
            type=bool,
            help=f"Show img while model iterations",
            default=False,
            dest="show_step",
        )
        dd_parser.add_argument(
            "-d",
            "--directory",
            type=str,
            help="Model directory",
            default=f"deepdream_dir/",
            dest="directory",
        )
        dd_parser.add_argument(
            "-f",
            "--filename",
            type=str,
            help=f"Image filename [if a filename is provided, the file is saved]",
            default=None,
            dest="filename",
        )

    def _add_subparser_args(self, parser):
        super()._add_subparser_args(parser)
        subparser = parser.add_subparsers(help="DeepDream")
        dd_parser = subparser.add_parser(name="DD")
        dd_parser.add_argument(
            "-cp",
            "--content_path",
            type=str,
            help="Content path of the content image",
            dest="content_path",
        )
        self._options_args(dd_parser)
        self._model_args(dd_parser)

    @staticmethod
    def _random_roll(img, maxroll):
        shift = tf.random.uniform(
            shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32
        )
        img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
        return shift, img_rolled

    @staticmethod
    def _calc_loss(img, model):
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = model(img_batch)
        if len(layer_activations) == 1:
            layer_activations = [layer_activations]

        losses = [tf.math.reduce_mean(act) for act in layer_activations]
        return tf.reduce_sum(losses)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
        )
    )
    def _exec(self, img, tile_size=512):
        """
        https://www.tensorflow.org/guide/function
        """
        shift, img_rolled = self._random_roll(img, tile_size)
        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)
        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[x : x + tile_size, y : y + tile_size]
                    loss = self._calc_loss(img_tile, self.model)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])
        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        return gradients

    def __init__(self):
        self.m_ = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet"
        )
        super().__init__(prog="DeepDream")
        os.makedirs(self.args.directory)
        layers = [self.m_.get_layer(name).output for name in self.args.layers]
        self.model = tf.keras.Model(inputs=self.m_.input, outputs=layers)

    def _run(
        self,
        img,
        octaves: list,
        step_size: float,
        octave_scale: int,
        steps_per_octave: int,
    ):
        base_shape = tf.shape(img)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)

        initial_shape = img.shape[:-1]
        img = tf.image.resize(img, initial_shape)
        for octave in octaves:
            start = time.time()
            # Scale the image based on the octave
            new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (
                octave_scale ** octave
            )
            img = tf.image.resize(img, tf.cast(new_size, tf.int32))

            for step in range(steps_per_octave):
                gradients = self._exec(img)
                img = img + gradients * step_size
                img = tf.clip_by_value(img, -1, 1)
            if self.args.show_step:
                plt.imshow(img)
                plt.show()
            logging.info(f"Octave {octave} - time: {time.time() - start:.2f}s")
        result = self.normalize_img(img)
        return result

    def run(self, img_path: str = None):
        if img_path is None:
            img_path = self.args.content_path
        original_img = tf.constant(self.get_img(img_path))
        dd_img = self._run(
            img=original_img,
            octaves=self.args.octaves,
            step_size=self.args.step_size,
            octave_scale=self.args.octave_scale,
            steps_per_octave=self.args.epochs,
        )
        if self.args.filename is not None:
            dd_img = tf.image.resize(dd_img, tf.shape(original_img)[:-1])
            dd_img = tf.image.convert_image_dtype(dd_img / 255.0, dtype=tf.uint8)
            PIL.Image.fromarray(np.array(dd_img)).save(
                f"{self.args.directory}/{self.args.filename}.png"
            )
