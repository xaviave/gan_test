import os
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display as display

from srcs.ImageHandler import ImageHandler


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}


class StyleTransfer(ImageHandler):
    epochs = 30
    style_weight = 1e-2
    content_weight = 1e4
    steps_per_epoch = 100
    total_variation_weight = 30

    # sample tf
    content_layers = ["block5_conv2"]
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    sample_content_path = tf.keras.utils.get_file(
        "YellowLabradorLooking_new.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
    )
    sample_style_path = tf.keras.utils.get_file(
        "kandinsky5.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
    )

    @staticmethod
    def _high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
        return x_var, y_var

    def _style_content_loss(self, outputs):
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]
        style_loss = tf.add_n(
            [
                tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2)
                for name in style_outputs.keys()
            ]
        )
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n(
            [
                tf.reduce_mean(
                    (content_outputs[name] - self.content_targets[name]) ** 2
                )
                for name in content_outputs.keys()
            ]
        )
        content_loss *= self.content_weight / self.num_content_layers
        loss = style_loss + content_loss
        return loss

    @staticmethod
    def _clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def _total_variation_loss(self, image):
        x_deltas, y_deltas = self._high_pass_x_y(image)
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

    @tf.function()
    def train_step(self, image, save: bool = False):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self._style_content_loss(outputs)
            loss += self.total_variation_weight * tf.image.total_variation(image)
        # if save:
        #
        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(self._clip_0_1(image))

    def _train(self, image):
        step = 0
        start = time.time()
        for n in range(self.epochs):
            start_e = time.time()
            for m in range(self.steps_per_epoch):
                step += 1
                self.train_step(image)
                print(".", end="")
            self.tensor_to_image(image).save(f"{self.m_path}/row_image_step_{step}.png")
            print(f"\nTrain step: {step} | time epoch: {time.time() - start_e}")
        end = time.time()
        print(f"Total time: {end - start:.1f}")

    def _init_style_content(
        self,
        content_path: str,
        style_path: str,
        style_layers: list = style_layers,
        content_layers: list = content_layers,
    ):
        self.style_image = self.load_img(style_path)
        self.content_image = self.load_img(content_path)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_style_layers = len(self.style_layers)
        self.num_content_layers = len(self.content_layers)

    def _init_nn(self):
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        self.vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")

    def _init_target(self):
        self.extractor = StyleContentModel(self.style_layers, self.content_layers)
        self.style_targets = self.extractor(self.style_image)["style"]
        self.content_targets = self.extractor(self.content_image)["content"]

    def __init__(
        self,
        m_name: str,
        content_path: str = sample_content_path,
        style_path: str = sample_style_path,
    ):
        super().__init__()
        self.m_path = f"images/{m_name}"
        os.makedirs(self.m_path)
        self.content_path = content_path
        self._init_style_content(content_path, style_path)
        self._init_nn()
        self._init_target()

    def run(self, ret: bool = False, save: bool = True):
        image = tf.Variable(self.content_image)
        self._train(image)
        self._total_variation_loss(image).numpy()
        tf.image.total_variation(image).numpy()
        if save:
            self.tensor_to_image(image).save(f"{self.m_path}/stylized-image.png")
            print(self.m_path, self.content_path)
            self.save_gif(self.m_path, self.content_path)
        if ret:
            return image
