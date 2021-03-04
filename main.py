from srcs.GanHandler import GanHandler
from srcs.StyleTransfer import StyleTransfer


def run_mnist_gan():
    gan = GanHandler()
    gan.train(gan.dataset)


def run_style_nn():
    s = StyleTransfer()
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
    - 
    """
    )
    # run_mnist_gan()
    run_style_nn()
