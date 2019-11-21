 require_relative "../base_test"

module Tensorflow
  module Data
    class MapDatasetTest < BaseTest
      extend Decorator

      def three_components
        columns = 7
        [Numo::Int32.new(columns).seq,
         Numo::NArray[[1, 2, 3]] * Numo::Int32.new(columns).seq[true, :new],
         Numo::NArray[37.0] * Numo::Int32.new(columns).seq]
      end

      def function_no_parameters
        @function_no_parameters ||=
            Graph::Graph.new.as_default do |func_graph|
              const = Tensorflow.constant(42)
              func_graph.to_function('function_no_parameters', nil, [], [const])
            end
      end

      def test_function_no_parameter
        self.eager_and_graph do |context|
          components = Numo::NArray[[1, 2, 3]]
          dataset = TensorDataset.new(components)

          ExecutionContext.current.add_function(self.function_no_parameters)
          map_dataset = MapDataset.new(dataset, self.function_no_parameters)

          result = self.result(context, map_dataset)
          assert_equal([42], result)
        end
      end

      def function_one_parameter
        @function_one_parameter ||=
          Graph::Graph.new.as_default do |func_graph|
            x = Tensorflow.placeholder(:int32, name: "x")
            square = Math.square(x)
            func_graph.to_function('function_one_parameter', nil, [x], [square])
          end
      end

      def test_function_one_parameter
        self.eager_and_graph do |context|
          components = [[1, 2, 3]]
          dataset = TensorDataset.new(components)

          ExecutionContext.current.add_function(self.function_one_parameter)

          map_dataset = MapDataset.new(dataset, self.function_one_parameter)
          result = self.result(context, map_dataset)

          result.each_with_index do |record, index|
            assert_equal(components[0][0] ** 2, record[0])
            assert_equal(components[0][1] ** 2, record[1])
            assert_equal(components[0][2] ** 2, record[2])
          end
        end
      end

      @tf.function(input_signature=[[:int32]])
      def one_parameter(x)
        Math.square(x)
      end

      def test_one_parameter
        self.eager_and_graph do |context|
          components = [[1, 2, 3]]
          dataset = TensorDataset.new(components)

          map_dataset = MapDataset.new(dataset, self.one_parameter)
          result = self.result(context, map_dataset)

          result.each_with_index do |record, index|
            assert_equal(components[0][0] ** 2, record[0])
            assert_equal(components[0][1] ** 2, record[1])
            assert_equal(components[0][2] ** 2, record[2])
          end
        end
      end

      def function_three_parameters
        @function_three_parameters ||=
            Graph::Graph.new.as_default do |func_graph|
              x = Tensorflow.placeholder(:int32)
              y = Tensorflow.placeholder(:int32)
              z = Tensorflow.placeholder(:double)

              r1 = Math.square(x)
              r2 = Math.square(y)
              r3 = Math.square(z)
              func_graph.to_function('function_three_parameters', nil, [x, y, z], [r1, r2, r3])
            end
      end

      def test_function_three_parameters
        self.eager_and_graph do |context|
          dataset = TensorSliceDataset.new(self.three_components)
          dataset = TensorSliceDataset.new(self.three_components)

          ExecutionContext.current.add_function(self.function_three_parameters)

          map_dataset = MapDataset.new(dataset, self.function_three_parameters)
          result = self.result(context, map_dataset)

          result.each_with_index do |record, index|
            assert_equal(self.three_components[0][index] ** 2, record[0])
            assert_equal(self.three_components[1][index, true] ** 2, record[1])
            assert_equal(self.three_components[2][index] ** 2, record[2])
          end
        end
      end

      @tf.function(input_signature=[[:int32], [:int32], [:double]])
      def three_parameters(x, y, z)
        [Math.square(x), Math.square(y), Math.square(z)]
      end

      def test_three_parameters
        self.eager_and_graph do |context|
          dataset = TensorSliceDataset.new(self.three_components)
          map_dataset = MapDataset.new(dataset, self.three_parameters)
          result = self.result(context, map_dataset)

          result.each_with_index do |record, index|
            assert_equal(self.three_components[0][index] ** 2, record[0])
            assert_equal(self.three_components[1][index, true] ** 2, record[1])
            assert_equal(self.three_components[2][index] ** 2, record[2])
          end
        end
      end

      @tf.function([[:string]])
      def change_output_type_and_shape(value)
        Tf.reshape(Tf.cast(value, :int32), [2, 2])
      end

      def test_change_output_type_and_shape
        components = Numo::NArray[['1', '2', '3', '4']]
        dataset = TensorSliceDataset.new(components)
        assert_equal([["1", "2", "3", "4"]], dataset.data)
        assert_equal([:string], dataset.output_types)
        assert_equal([[4]], dataset.output_shapes)

        map_dataset = dataset.map_func(change_output_type_and_shape)
        assert_equal([:int32], map_dataset.output_types)
        assert_equal([[4]], dataset.output_shapes)
      end
    end
  end
end
