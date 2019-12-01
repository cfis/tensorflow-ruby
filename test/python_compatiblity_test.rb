require_relative "base_test"

module Tensorflow
  class PythonCompatiblityTest < BaseTest
    def test_tf
      assert_same(Tensorflow, Tf)
    end

    def test_set_mode
      Tensorflow.disable_eager_execution
      assert_equal(Tensorflow::GRAPH_MODE, Tensorflow.execution_mode)

      Tensorflow.enable_eager_execution
      assert_equal(Tensorflow::EAGER_MODE, Tensorflow.execution_mode)
    end

    def test_global_variables
      Graph::Graph.new.as_default do |graph|
        v = Variable.new([1, 2], name: "v")
        w = Variable.new([3, 4], name: "w")

        variables = Tensorflow.global_variables.to_a
        assert_equal(2, variables.length)
        assert_equal(v, variables[0])
        assert_equal(w, variables[1])
      end
    end

    def test_global_variables_initializer
      Graph::Graph.new.as_default do |graph|
        v = Variable.new([1, 2], name: "v")
        w = Variable.new([3, 4], name: "w")

        operation = Tensorflow.global_variables_initializer
        assert_kind_of(Graph::Operation, operation)
        assert_equal(2, operation.num_control_inputs)
        assert_equal(v.handle, operation.control_inputs[0])
        assert_equal(w.handle, operation.control_inputs[1])
      end
    end
  end
end