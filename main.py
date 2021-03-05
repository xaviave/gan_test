import os
import time
import random

from srcs.DeepDream import DeepDream
from srcs.GanHandler import GanHandler
from srcs.StyleTransfer import StyleTransfer


def run_mnist_gan():
    gan = GanHandler()
    gan.train(gan.dataset)


def run_style_nn():
    contents_list = os.listdir("contents")
    styles_list = os.listdir("styles")
    for i in range(1):
        r_content = random.choices(contents_list)[0]
        r_style = random.choices(styles_list)[0]
        s = StyleTransfer(
            m_name=f"{r_style[:-4]}_{i}_{time.time()}",
            content_path=f"contents/{r_content}",
            style_path=f"styles/{r_style}",
        )
        s.run()


def run_deepdream():
    deepdream = DeepDream()
    deepdream.run("contents/20201102_142043.png")


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
