import glob
import PIL.Image

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from natsort import natsorted
from srcs.tools.ImageEnhancer import ImageEnhancer


class ImageHandler(ImageEnhancer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_img(img_path):
        img = PIL.Image.open(img_path).convert("RGB")
        return np.array(img)

    @staticmethod
    def normalize_img(img):
        img = 255 * (img + 1.0) / 2.0
        return tf.cast(img, tf.uint8)

    @staticmethod
    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    @staticmethod
    def imshow(image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)
        plt.imshow(image)
        if title:
            plt.title(title)

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def save_gif(self, path):
        anim_file = f"{path}/images.gif"
        filenames = glob.glob(f"{path}/*.png")
        for x in filenames:
            self.enhance_image(x)
        img, *imgs = [PIL.Image.open(f) for f in natsorted(filenames)]
        img.save(
            fp=anim_file,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=50,
            loop=0,
        )
