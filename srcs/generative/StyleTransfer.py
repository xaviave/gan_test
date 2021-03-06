import os
import time
import logging

import tensorflow as tf

from srcs.args.ArgParser import ArgParser
from srcs.tools.ImageHandler import ImageHandler


class StyleTransfer(ArgParser, ImageHandler):
    """
    Use a Style image and applied it on the Content image
    by getting details and colors from VGG19 layers.

    Parser args define all the model hyperparameters:
        - epochs
        - batch_size
        - style_layers
        - style_weight
        - content_layers
        - content_weight
        - variation_weight
    """

    def _model_args(self, st_parser):
        st_parser.add_argument(
            "-e",
            "--epochs",
            type=int,
            help=f"Number of epoch",
            default=10,
            dest="epochs",
        )
        st_parser.add_argument(
            "-bs",
            "--batch_size",
            type=int,
            help=f"Batch size in epoch",
            default=150,
            dest="batch_size",
        )
        st_parser.add_argument(
            "-sl",
            "--style_layers",
            type=str,
            nargs="+",
            help=f"StyleTransfer model is created based VGG19 model's layers | List of Layers (style) used in model: [{' - '.join([l.name for l in self.m_.layers])}]",
            default=[
                "block1_conv1",
                "block2_conv1",
                "block3_conv1",
                "block4_conv1",
                "block5_conv1",
            ],
            dest="style_layers",
        )
        st_parser.add_argument(
            "-sw",
            "--style_weight",
            type=float,
            help=f"Weight applied on the style image",
            default=1e-2,
            dest="style_weight",
        )
        st_parser.add_argument(
            "-cl",
            "--content_layers",
            type=str,
            nargs="+",
            help=f"StyleTransfer model is created based VGG19 model's layers | List of Layers (content) used in model: [{' - '.join([l.name for l in self.m_.layers])}]",
            default=["block5_conv2"],
            dest="content_layers",
        )
        st_parser.add_argument(
            "-cw",
            "--content_weight",
            type=float,
            help=f"Weight applied on the content image",
            default=1e4,
            dest="content_weight",
        )
        st_parser.add_argument(
            "-vw",
            "--variation_weight",
            type=int,
            help=f"Variation weight between content and style",
            default=50,
            dest="variation_weight",
        )

    @staticmethod
    def _options_args(st_parser):
        st_parser.add_argument(
            "-ss",
            "--show_step",
            type=bool,
            help=f"Show img while model iterations",
            default=False,
            dest="show_step",
        )
        st_parser.add_argument(
            "-d",
            "--directory",
            type=str,
            help="Model directory",
            default=f"style_transfer_dir/",
            dest="directory",
        )
        st_parser.add_argument(
            "-f",
            "--filename",
            type=str,
            help=f"Image filename [if a filename is provided, the file is saved]",
            default=None,
            dest="filename",
        )

    def _add_subparser_args(self, parser):
        super()._add_subparser_args(parser)
        subparser = parser.add_subparsers(help="StyleTransfer")
        st_parser = subparser.add_parser(name="ST")
        st_parser.add_argument(
            "-sp",
            "--style_path",
            type=str,
            help="Style path of the style image",
            dest="style_path",
        )
        st_parser.add_argument(
            "-cp",
            "--content_path",
            type=str,
            help="Content path of the content image",
            dest="content_path",
        )
        self._model_args(st_parser)
        self._options_args(st_parser)

    @staticmethod
    def _high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
        return x_var, y_var

    def _style_content_loss(self, style_outputs, content_outputs):
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

    @staticmethod
    def gram_matrix(input_tensor):
        result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)

    def _exec(self, inputs, opt):
        """
        Expects float input in [0,1]
        """
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.model(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )

        style_outputs = [
            self.gram_matrix(style_output) for style_output in style_outputs
        ]
        if opt == "content":
            return {
                content_name: value
                for content_name, value in zip(
                    self.args.content_layers, content_outputs
                )
            }
        elif opt == "style":
            return {
                style_name: value
                for style_name, value in zip(self.args.style_layers, style_outputs)
            }

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            style_outputs = self._exec(image, "style")
            content_outputs = self._exec(image, "content")
            loss = self._style_content_loss(style_outputs, content_outputs)
            loss += self.args.variation_weight * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(self._clip_0_1(image))

    def _train(self, image):
        step = 0
        start = time.time()
        for n in range(self.args.epochs):
            start_e = time.time()
            for m in range(self.args.batch_size):
                step += 1
                self.train_step(image)
                print(".", end="")
            self.tensor_to_image(image).save(f"{self.m_path}/row_image_step_{step}.png")
            print(f"\nTrain step: {step} | time epoch: {time.time() - start_e}")
        end = time.time()
        print(f"Total time: {end - start:.1f}")

    def _init_nn(self):
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        self.m_ = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
        self.m_.trainable = False

    def _init_options(self, m_name):
        self.m_path = f"{self.args.directory}/{m_name}"
        logging.info(f"Using dir path: {self.m_path}")
        os.makedirs(self.m_path)

    def _init_style_content(self):
        self.style_image = self.load_img(self.args.style_path)
        self.content_image = self.load_img(self.args.content_path)
        self.content_layers = self.args.content_layers
        self.style_layers = self.args.style_layers
        self.num_style_layers = len(self.style_layers)
        self.num_content_layers = len(self.content_layers)

    def _init_target(self):
        self.model = tf.keras.Model(
            [self.m_.input],
            [
                self.m_.get_layer(n).output
                for n in self.args.content_layers + self.args.style_layers
            ],
        )
        self.style_targets = self._exec(self.style_image, "style")
        self.content_targets = self._exec(self.content_image, "content")

    def __init__(self, m_name: str):
        """
        https://www.tensorflow.org/tutorials/generative/style_transfer
        """
        self._init_nn()
        super().__init__()
        self._init_options(m_name)
        self._init_style_content()
        self._init_target()

    def run(self, ret: bool = False):
        image = tf.Variable(self.content_image)
        self._train(image)
        image = self.tensor_to_image(image)
        if self.args.filename is not None:
            image.save(f"{self.m_path}/{self.args.filename}.png")
        if ret:
            return image
