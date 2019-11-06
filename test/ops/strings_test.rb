require_relative "../test_helper"

class StringsTest < Minitest::Test
  def setup
    Tensorflow.execution_mode = Tensorflow::EAGER_MODE
  end

  def test_join
    assert_equal("helloworld", Tensorflow::Strings.join(["hello", "world"]).value)
    assert_equal("hello world", Tensorflow::Strings.join(["hello", "world"], separator: " ").value)
  end

  def test_length
    assert_equal(5, Tensorflow::Strings.length("hello").value)
  end

  def test_lower
    assert_equal("hello", Tensorflow::Strings.lower("HELLO").value)
  end

  def test_to_number
    assert_equal(123, Tensorflow::Strings.to_number("123").value)
  end

  def test_strip
    assert_equal("hello", Tensorflow::Strings.strip(" hello ").value)
  end

  def test_upper
    assert_equal("HELLO", Tensorflow::Strings.upper("hello").value)
  end
end
