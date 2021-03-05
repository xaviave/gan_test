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
    contents_list = os.listdir("contents")
    for i in range(20):
        r_content = random.randrange(len(contents_list))
        deepdream.run(contents_list[r_content])


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
