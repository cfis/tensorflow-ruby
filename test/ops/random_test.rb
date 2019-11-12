require_relative "../test_helper"

class RandomTest < Minitest::Test
  def setup
    Tensorflow.execution_mode = Tensorflow::EAGER_MODE
  end

  def test_normal
    result = Tensorflow::Random.normal([2, 2], stddev: 10.0)
    assert_equal([2, 2], result.shape)
    assert_equal(:float, result.dtype)
  end

  def test_uniform
    result = Tensorflow::Random.uniform([2, 2])
    assert_equal([2, 2], result.shape)
    assert_equal(:float, result.dtype)
  end
end
