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
    for i in range(1):
        s = StyleTransfer(
            m_name=f"{time.time()}".replace(".", "_"),
        )
        s.run()


def run_deepdream():
    deepdream = DeepDream()
    os.mkdir(f"{deepdream.args.layers[0]}_{deepdream.args.layers[1]}")
    contents_list = os.listdir("contents")
    for i in range(20):
        deepdream.args.filename = (
            f"{deepdream.args.layers[0]}_{deepdream.args.layers[1]}/img_{i}"
        )
        r_content = contents_list[random.randrange(len(contents_list))]
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
    run_style_nn()
    # run_deepdream()
