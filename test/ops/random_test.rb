require_relative "../test_helper"

class RandomTest < Minitest::Test
  def test_normal
    result = Tf::Random.normal([2, 2], stddev: 10)
    assert_equal([2, 2], result.shape)
    assert_equal(:float, result.dtype)
    puts result
  end
end
