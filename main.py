import os
import random

from srcs.GanHandler import GanHandler
from srcs.StyleTransfer import StyleTransfer


def run_mnist_gan():
    gan = GanHandler()
    gan.train(gan.dataset)


def run_style_nn():
    contents_list = os.listdir("contents")
    styles_list = os.listdir("styles")
    for i in range(20):
        r_content = random.choices(contents_list)
        r_style = random.choices(styles_list)
        s = StyleTransfer(
            m_name=f"{r_style}_{i}",
            content_path=f"contents/{r_content}",
            style_path=f"styles/{r_style}",
        )
        s.run()


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
