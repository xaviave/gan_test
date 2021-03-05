import os
import time

import tensorflow as tf

from srcs.args.ArgParser import ArgParser
from srcs.tools.ImageHandler import ImageHandler


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
        """
        Expects float input in [0,1]
        """
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


class StyleTransfer(ArgParser, ImageHandler):
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

    def _model_args(self, parser):
        parser.add_argument(
            "-stvw",
            "--styletransfer_variation_weight",
            type=int,
            help=f"Variation weight between content and style",
            default=50,
            dest="variation_weight",
        )
        parser.add_argument(
            "-stcw",
            "--styletransfer_content_weight",
            type=float,
            help=f"Weight applied on the content image",
            default=1e4,
            dest="content_weight",
        )
        parser.add_argument(
            "-stsw",
            "--styletransfer_style_weight",
            type=float,
            help=f"Weight applied on the style image",
            default=1e-2,
            dest="style_weight",
        )
        parser.add_argument(
            "-ste",
            "--styletransfer_epochs",
            type=int,
            help=f"Number of epoch",
            default=10,
            dest="epochs",
        )
        parser.add_argument(
            "-stse",
            "--styletransfer_step_epochs",
            type=int,
            help=f"Number of steps in epoch",
            default=150,
            dest="step_in_epochs",
        )

    def _add_parser_args(self, parser):
        super()._add_parser_args(parser)
        parser.add_argument(
            "-stss",
            "--styletransfer_show_step",
            type=bool,
            help=f"Show img while model iterations",
            default=False,
            dest="show_step",
        )
        parser.add_argument(
            "-stsi",
            "--styletransfer_save_image",
            type=str,
            help=f"Image filename [if a filename is provided, the file is saved]",
            default=None,
            dest="img_name",
        )
        parser.add_argument(
            "-stls",
            "--styletransfer_layers",
            type=str,
            nargs="+",
            help=f"StyleTransfer model is created based VGG19 model's layers\nList of Layers in model: {' - '.join([l.name for l in self.m_.layers])}",
            default=["mixed3", "mixed5"],
            dest="layers",
        )
        self._model_args(parser)

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
        style_loss *= self.args.style_weight / self.num_style_layers

        content_loss = tf.add_n(
            [
                tf.reduce_mean(
                    (content_outputs[name] - self.content_targets[name]) ** 2
                )
                for name in content_outputs.keys()
            ]
        )
        content_loss *= self.args.content_weight / self.num_content_layers
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
            loss += self.args.total_variation_weight * tf.image.total_variation(image)
        # if save:
        #
        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(self._clip_0_1(image))

    def _train(self, image):
        step = 0
        start = time.time()
        for n in range(self.args.epochs):
            start_e = time.time()
            for m in range(self.args.steps_per_epoch):
                step += 1
                self.train_step(image)
                print(".", end="")
            self.tensor_to_image(image).save(f"{self.m_path}/row_image_step_{step}.png")
            print(f"\nTrain step: {step} | time epoch: {time.time() - start_e}")
        end = time.time()
        print(f"Total time: {end - start:.1f}")

    def _init_style_content(
        self,
        style_path: str,
        content_path: str,
    ):
        self.style_image = self.load_img(style_path)
        self.content_image = self.load_img(content_path)
        self.content_layers = self.args.content_layers
        self.style_layers = self.args.style_layers
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
        """
        https://www.tensorflow.org/tutorials/generative/style_transfer
        """
        super().__init__()
        self.m_path = f"images/{m_name}"
        os.makedirs(self.m_path)
        self.content_path = content_path
        self._init_style_content(style_path, content_path)
        self._init_nn()
        self._init_target()

    def run(self, ret: bool = False, save: bool = True):
        image = tf.Variable(self.content_image)
        self._train(image)
        if save:
            self.tensor_to_image(image).save(f"{self.m_path}/stylized-image.png")
        if ret:
            return image
