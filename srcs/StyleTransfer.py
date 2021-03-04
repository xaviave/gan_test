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
    epochs = 10
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
    content_path = tf.keras.utils.get_file(
        "YellowLabradorLooking_new.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
    )
    style_path = tf.keras.utils.get_file(
        "kandinsky5.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
    )

    @staticmethod
    def high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
        return x_var, y_var

    def style_content_loss(self, outputs):
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

    def clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def total_variation_loss(self, image):
        x_deltas, y_deltas = self.high_pass_x_y(image)
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)
            loss += self.total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(self.clip_0_1(image))

    def _train(self, image):
        step = 0
        start = time.time()
        for n in range(self.epochs):
            for m in range(self.steps_per_epoch):
                step += 1
                self.train_step(image)
                print(".", end="")
            ImageHandler().tensor_to_image(image).save(f"row_image_step_{step}.png")
            print(f"Train step: {step}")
        end = time.time()
        print(f"Total time: {end - start:.1f}")

    def _init_style_content(
        self,
        content_path: str = content_path,
        style_path: str = style_path,
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

    def __init__(self):
        self._init_style_content()
        self._init_nn()

    def show_art(self, content_image, fig, pos):
        x_deltas, y_deltas = self.high_pass_x_y(content_image)
        plt.figure(figsize=(14, 10))
        plt.subplot(*fig, pos[0]["id"])
        ImageHandler().imshow(self.clip_0_1(2 * y_deltas + 0.5), pos[0]["com"])
        plt.subplot(*fig, pos[1]["id"])
        ImageHandler().imshow(self.clip_0_1(2 * x_deltas + 0.5), pos[1]["com"])

    def show_art_sobel(self, content_image, fig, pos):
        sobel = tf.image.sobel_edges(content_image)
        plt.figure(figsize=(14, 10))
        plt.subplot(*fig, pos[0]["id"])
        ImageHandler().imshow(
            self.clip_0_1(sobel[..., 0] / 4 + 0.5), "Horizontal Sobel-edges"
        )
        plt.subplot(1, 2, 2)
        ImageHandler().imshow(
            self.clip_0_1(sobel[..., 1] / 4 + 0.5), "Vertical Sobel-edges"
        )

    def run(self):
        self.extractor = StyleContentModel(self.style_layers, self.content_layers)
        self.style_targets = self.extractor(self.style_image)["style"]
        self.content_targets = self.extractor(self.content_image)["content"]
        image = tf.Variable(self.content_image)
        self._train(image)
        self.show_art(
            self.content_image,
            [2, 2],
            [
                {"id": 1, "com": "Horizontal Deltas: Original"},
                {"id": 2, "com": "Vertical Deltas: Original"},
            ],
        )
        self.show_art(
            image,
            [2, 2],
            [
                {"id": 3, "com": "Horizontal Deltas: Styled"},
                {"id": 4, "com": "Vertical Deltas: Styled"},
            ],
        )

        self.show_art_sobel(
            self.content_image,
            [1, 2],
            [
                {"id": 1, "com": "Horizontal Sobel-edges"},
                {"id": 2, "com": "Vertical Sobel-edges"},
            ],
        )
        plt.subplot(1, 2, 1)
        plt.show()

        self.total_variation_loss(image).numpy()
        tf.image.total_variation(image).numpy()

        image = tf.Variable(self.content_image)
        file_name = "stylized-image.png"
        self.opt.apply_gradients()
        ImageHandler().tensor_to_image(image).save(file_name)
