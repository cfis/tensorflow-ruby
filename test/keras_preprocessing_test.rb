require_relative "base_test"

module Tensorflow
  class KerasPreprocessingTest < BaseTest
    def setup
      Tensorflow.execution_mode = Tensorflow::EAGER_MODE
    end

    def test_image
      file = Tensorflow::Keras::Utils.get_file("grace_hopper.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
      img = Tensorflow::Keras::Preprocessing::Image.load_img(file, target_size: [224, 224])
      x = Tensorflow::Keras::Preprocessing::Image.img_to_array(img)
      assert_equal [224, 224, 3], x.shape
    end
  end
end