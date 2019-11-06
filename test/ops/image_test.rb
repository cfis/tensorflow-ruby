require_relative "../test_helper"

class ImageTest < Minitest::Test
  def setup
    Tensorflow.execution_mode = Tensorflow::EAGER_MODE
  end

  def test_decode_jpeg
    assert_equal([227, 320, 3], Tensorflow::Image.decode_jpeg(jpeg_contents).shape)
  end

  def test_is_jpeg
    assert(Tensorflow::Image.is_jpeg(jpeg_contents).value)
    assert(Tensorflow::IO.is_jpeg(jpeg_contents).value)
    assert_equal(0, Tensorflow::Image.is_jpeg("notjpeg").value)
    assert(0, Tensorflow::IO.is_jpeg("notjpeg").value)
  end

  def test_resize
    image = Tensorflow::Image.decode_jpeg(jpeg_contents)
    assert_equal([192, 192, 3], Tensorflow::Image.resize(image, [192, 192]).shape)
  end

  private

  def jpeg_contents
    Tensorflow::IO.read_file("test/support/bears.jpg")
  end
end
