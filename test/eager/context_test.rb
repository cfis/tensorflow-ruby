require_relative "../test_helper"

module Tensorflow
  module Eager
    class ContextTest < Minitest::Test
      def test_works
        context = Eager::Context.default
        assert !context.function?("hi")
        assert_equal :silent, context.device_policy
        context.enable_run_metadata
        context.disable_run_metadata
      end
    end
  end
end
