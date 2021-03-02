from srcs.GanHandler import GanHandler


def run_mnist_gan():
    gan = GanHandler()
    gan.train()


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
    run_mnist_gan()
