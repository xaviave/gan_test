import logging
import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


class ImageEnhancer:
    SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

    @staticmethod
    def _preprocess_image(hr_image: np.array):
        """
        Loads image from path and preprocesses to make it model ready
        Args:
          image_path: Path to the image file
        """
        # If PNG, remove the alpha channel. The model only supports
        # images with 3 color channels.
        if hr_image.shape[-1] == 4:
            hr_image = hr_image[..., :-1]
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)

    @staticmethod
    def save_image(image: np.array, filename: str):
        """
        Saves unscaled Tensor Images.
        Args:
          image: 3D image tensor. [height, width, channels]
          filename: Name of the file to save to.
        """
        if not isinstance(image, Image.Image):
            image = tf.clip_by_value(image, 0, 255)
            image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        image.save(filename)
        print(f"Saved as {filename}")

    @staticmethod
    def plot_image(image:np.array, title:str=""):
        """
        Plots images from image tensors.
        Args:
          image: 3D image tensor. [height, width, channels].
          title: Title to display in the plot.
        """
        image = np.asarray(image)
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        plt.imshow(image)
        plt.axis("off")
        plt.title(title)

    @staticmethod
    def downscale_image(image: np.array):
        """
        Scales down images using bicubic downsampling.
        Args:
            image: 3D or 4D tensor of preprocessed image
        """
        image_size = []
        if len(image.shape) == 3:
            image_size = [image.shape[1], image.shape[0]]
        else:
            raise ValueError("Dimension mismatch. Can work only on single image.")

        image = tf.squeeze(tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8))

        lr_image = np.asarray(
            Image.fromarray(image.numpy()).resize(
                [image_size[0] // 4, image_size[1] // 4], Image.BICUBIC
            )
        )

        lr_image = tf.expand_dims(lr_image, 0)
        lr_image = tf.cast(lr_image, tf.float32)
        return lr_image

    def __init__(self):
        os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
        self.model = hub.load(self.SAVED_MODEL_PATH)

    def enhance_image(self, image_path: str = None, raw_image: np.array = None, ret: bool = False, save: bool = False):
        """
        https://www.tensorflow.org/hub/tutorials/image_enhancing
        """
        if image_path is not None:
            raw_image = tf.image.decode_image(tf.io.read_file(image_path))

        try:
            hr_image = self._preprocess_image(raw_image)
        except AttributeError:
            print(f"Must provide an 'image_path' or a numpy array")
            exit()
        # hr_image = self.downscale_image(tf.squeeze(hr_image))
        start = time.time()
        fake_image = self.model(tf.cast(hr_image, tf.float32))
        fake_image = tf.squeeze(fake_image)
        tf.cast(fake_image, tf.uint8)
        logging.info(f"Time Taken: {time.time() - start}")
        if save:
            self.save_image(fake_image, filename=image_path)
        if ret:
            return fake_image
