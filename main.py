from srcs.GanHandler import GanHandler
from srcs.StyleTransfer import StyleTransfer


def run_mnist_gan():
    gan = GanHandler()
    gan.train(gan.dataset)


def run_style_nn():
    s = StyleTransfer(m_name="default")
    s.run()
    s = StyleTransfer(m_name="style_1", style_path="styles/style_1.jpg")
    s.run()
    s = StyleTransfer(m_name="style_2", style_path="styles/style_2.jpg")
    s.run()
    s = StyleTransfer(m_name="style_3", style_path="styles/style_3.jpg")
    s.run()
    s = StyleTransfer(m_name="style_4", style_path="styles/style_4.jpg")
    s.run()
    s = StyleTransfer(m_name="style_5", style_path="styles/style_5.jpg")
    s.run()
    s = StyleTransfer(m_name="style_6", style_path="styles/style_6.jpg")
    s.run()
    s = StyleTransfer(m_name="style_7", style_path="styles/style_7.jpg")
    s.run()
    s = StyleTransfer(m_name="style_8", style_path="styles/style_8.jpg")
    s.run()
    s = StyleTransfer(m_name="style_9", style_path="styles/style_9.jpg")
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
