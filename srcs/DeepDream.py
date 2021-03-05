import cv2
import time
import PIL.Image

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from srcs.ImageHandler import ImageHandler


class DeepDream(ImageHandler):
    """
    https://www.tensorflow.org/tutorials/generative/deepdream
    """

    layers_names = ["mixed3", "mixed5"]

    @staticmethod
    def _random_roll(img, maxroll):
        shift = tf.random.uniform(
            shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32
        )
        img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
        return shift, img_rolled

    @staticmethod
    def _calc_loss(img, model):
        # Pass forward the image through the model to retrieve the activations.
        # Converts the image into a batch of size 1.
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = model(img_batch)
        if len(layer_activations) == 1:
            layer_activations = [layer_activations]

        losses = []
        for act in layer_activations:
            loss = tf.math.reduce_mean(act)
            losses.append(loss)

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
        super().__init__()
        m_ = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
        layers = [m_.get_layer(name).output for name in self.layers_names]
        self.model = tf.keras.Model(inputs=m_.input, outputs=layers)

    def run_deep_dream_with_octaves(
        self,
        img,
        steps_per_octave=50,
        step_size=0.01,
        octaves=range(-2, 3),
        octave_scale=1.3,
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

                if step % 10 == 0:
                    # save point
                    print(f"Octave {octave} | Step {step}")
            print(time.time() - start)
        result = self.normalize_img(img)
        plt.imshow(result)
        plt.show()
        return result

    def run(self, img_path):
        original_img = tf.constant(self.get_img(img_path))
        base_shape = tf.shape(original_img)[:-1]
        img = self.run_deep_dream_with_octaves(img=original_img, step_size=0.01)
        plt.imshow(img)
        plt.show()
        img = tf.image.resize(img, base_shape)
        img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
        PIL.Image.fromarray(np.array(img)).save(f"{time.time()}.png")
