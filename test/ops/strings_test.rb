require_relative "../base_test"

module Tensorflow
  class StringsTest < BaseTest
    def test_join
      self.eager_and_graph do |context|
        op = Strings.join(["hello", "world"])
        result = self.evaluate(op)
        assert_equal("helloworld", result)

        op = Strings.join(["hello", "world"], separator: " ")
        result = self.evaluate(op)
        assert_equal("hello world", result)
      end
    end

    def test_length
      self.eager_and_graph do |context|
        op = Strings.length("hello")
        result = self.evaluate(op)
        assert_equal(5, result)
      end
    end

    def test_lower
      self.eager_and_graph do |context|
        op = Strings.lower("HELLO")
        result = self.evaluate(op)
        assert_equal("hello", result)
      end
    end

    def test_to_number
      self.eager_and_graph do |context|
        op = Strings.to_number("123")
        result = self.evaluate(op)
        assert_equal(123, result)
      end
    end

    def test_strip
      self.eager_and_graph do |context|
        op = Strings.strip(" hello ")
        result = self.evaluate(op)
        assert_equal("hello", result)
      end
    end

    def test_upper
      self.eager_and_graph do |context|
        op = Strings.upper("hello")
        result = self.evaluate(op)
        assert_equal("HELLO", result)
      end
    end
  end
end
