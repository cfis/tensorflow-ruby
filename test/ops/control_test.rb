require_relative "../test_helper"

module Tensorflow
  class ControlTest < Minitest::Test
    def setup
      Tensorflow.disable_eager_execution
    end

    def test_group
      Graph::Graph.new.as_default do |graph|
        v = Variable.new([1, 2], name: "v")
        w = Variable.new([3, 4], name: "w")

        group = Control.group([v, w])
        assert_equal('NoOp', group.op_type)
        assert_equal(2, group.num_control_inputs)
        assert_equal(v.handle, group.control_inputs[0])
        assert_equal(w.handle, group.control_inputs[1])
      end
    end
  end
end
