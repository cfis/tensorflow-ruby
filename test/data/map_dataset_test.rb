require_relative "../test_helper"

module Tensorflow
  module Data
    class MapDatasetTest < Minitest::Test
      def test_simple
        components = [[1, 2, 3]]
        dataset = TensorDataset.new(components)

        func_graph = Graph::Graph.new
        x = func_graph.placeholder("x")

        op_desc = Graph::OperationDescription.new(func_graph, 'Square', [], name: 'square')
        op_desc.add_input(x)
        square = op_desc.save

        function = func_graph.to_function('MyFunc', nil, [x], [square], ['out1'])
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
