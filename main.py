import os
import time
import random

from srcs.generative.DeepDream import DeepDream
from srcs.generative.GanHandler import GanHandler
from srcs.generative.StyleTransfer import StyleTransfer


def run_mnist_gan():
    gan = GanHandler()
    gan.train(gan.dataset)


def run_style_nn():
    contents_list = os.listdir("contents")
    styles_list = os.listdir("styles")
    for i in range(1):
        r_content = random.randrange(len(contents_list))
        r_style = random.randrange(len(styles_list))
        s = StyleTransfer(
            m_name=f"{r_style[:-4]}_{i}_{time.time()}",
            content_path=f"contents/{contents_list[r_content]}",
            style_path=f"styles/{styles_list[r_style]}",
        )
        s.run()


def run_deepdream():
    deepdream = DeepDream()
    layers_list = deepdream.m_.layers
    contents_list = os.listdir("contents")
    for i in range(20):
        a = random.randint(2, len(layers_list) // 3)
        r_content = contents_list[random.randrange(len(contents_list))]
        deepdream.args.octaves = range(random.randint(-5, 5), random.randint(-5, 5))
        deepdream.args.step_size = random.uniform(0.01, 0.99)
        deepdream.args.octave_scale = random.uniform(0.5, 3.0)
        deepdream.args.steps_per_octave = random.randint(20, 200)
        deepdream.args.layers = [layers_list[random.randrange(len(contents_list))].name for _ in a]
        deepdream.args.img_name = f"{r_content}-{deepdream.args.octaves}-{deepdream.args.step_size}-{deepdream.args.octave_scale}-{deepdream.args.steps_per_octave}-{deepdream.args.layers}".replace(
            ".", "_"
        ).replace(
            " ", ""
        )
        deepdream.run(f"contents/{r_content}")


if __name__ == "__main__":
    print(
        """
check:
	- Tensor definition
	- tf.data.Dataset.from_tensor_slices
	- tf.train.Checkpoint
	- tf.GradientTape
	- tensorflow_hub
	"""
    )
    # run_mnist_gan()
    # run_style_nn()
    run_deepdream()
