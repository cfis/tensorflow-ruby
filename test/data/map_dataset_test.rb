require_relative "../test_helper"

module Tensorflow
  module Data
    class MapDatasetTest < Minitest::Test
      def setup
        Tensorflow.execution_mode = Tensorflow::EAGER_MODE
      end

      def test_simple
        components = [[1, 2, 3]]
        dataset = TensorDataset.new(components)

        function = Graph::Graph.new.as_default do |func_graph|
          x = Tensorflow.placeholder("x")
          square = Math.square(x)
          func_graph.to_function('MyFunc', nil, [x], [square], ['out1'])
        end

        Eager::Context.default.add_function(function)

        map_dataset = MapDataset.new(dataset, function, output_types: [:int32], output_shapes: [[2, 3]])

        map_dataset.each_with_index do |slice, i|
          assert_equal(components[0][0] ** 2, slice.value[0])
          assert_equal(components[0][1] ** 2, slice.value[1])
          assert_equal(components[0][2] ** 2, slice.value[2])
        end
      end
    end
  end
end
