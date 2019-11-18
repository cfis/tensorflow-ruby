require_relative "../test_helper"

module Tensorflow
  module Data
    class MapDatasetTest < Minitest::Test
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

      def test_function_no_parameters_eager
        Eager::Context.default.as_default do |context|
          components = Numo::NArray[[1, 2, 3]]
          dataset = TensorDataset.new(components)

          ExecutionContext.current.add_function(self.function_no_parameters)
          map_dataset = MapDataset.new(dataset, self.function_no_parameters)
          assert_equal([42], map_dataset.data)
        end
      end

      def test_function_no_parameters_graph
        Graph::Graph.new.as_default do |graph|
          components = [[1, 2, 3]]
          dataset = TensorDataset.new(components)

          graph.add_function(self.function_no_parameters)
          map_dataset = MapDataset.new(dataset, self.function_no_parameters)#, output_types: [:int32], output_shapes: [[1]])
          iterator = map_dataset.make_one_shot_iterator
          next_element = iterator.get_next

          result = Graph::Session.run(graph) do |session|
                      session.run(next_element)
                    end

          assert_equal(42, result)
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

      def test_function_one_parameter_eager
        Eager::Context.default.as_default do |context|
          components = [[1, 2, 3]]
          dataset = TensorDataset.new(components)

          ExecutionContext.current.add_function(self.function_one_parameter)

          map_dataset = MapDataset.new(dataset, self.function_one_parameter)
          map_dataset.each_with_index do |record, index|
            assert_equal(components[0][0] ** 2, record.value[0])
            assert_equal(components[0][1] ** 2, record.value[1])
            assert_equal(components[0][2] ** 2, record.value[2])
          end
        end
      end

      def test_function_one_parameter_graph
        Graph::Graph.new.as_default do |graph|
          components = [[1, 2, 3]]
          dataset = TensorDataset.new(components)

          ExecutionContext.current.add_function(self.function_one_parameter)
          map_dataset = MapDataset.new(dataset, self.function_one_parameter)
          iterator = map_dataset.make_one_shot_iterator
          next_element = iterator.get_next

          result = Graph::Session.run(graph) do |session|
            session.run(next_element)
          end

          assert_equal(components[0][0] ** 2, result[0])
          assert_equal(components[0][1] ** 2, result[1])
          assert_equal(components[0][2] ** 2, result[2])
        end
      end

      @tf.function(input_signature=[[:int32]])
      def one_parameter(x)
        Math.square(x)
      end

      def test_one_parameter_eager
        Eager::Context.default.as_default do |context|
          components = [[1, 2, 3]]
          dataset = TensorDataset.new(components)

          map_dataset = MapDataset.new(dataset, self.one_parameter)
          map_dataset.each_with_index do |record, index|
            assert_equal(components[0][0] ** 2, record.value[0])
            assert_equal(components[0][1] ** 2, record.value[1])
            assert_equal(components[0][2] ** 2, record.value[2])
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

      def test_function_three_parameters_eager
        Eager::Context.default.as_default do |context|
          dataset = TensorSliceDataset.new(self.three_components)
          dataset = TensorSliceDataset.new(self.three_components)

          ExecutionContext.current.add_function(self.function_three_parameters)

          map_dataset = MapDataset.new(dataset, self.function_three_parameters)

          map_dataset.each_with_index do |record, index|
            assert_equal(self.three_components[0][index] ** 2, record[0].value)
            assert_equal(self.three_components[1][index, true] ** 2, record[1].value)
            assert_equal(self.three_components[2][index] ** 2, record[2].value)
          end
        end
      end

      def test_function_three_parameters_graph
        columns = 7
        Graph::Graph.new.as_default do |graph|
          dataset = TensorSliceDataset.new(self.three_components)
          ExecutionContext.current.add_function(self.function_three_parameters)

          map_dataset = MapDataset.new(dataset, self.function_three_parameters)
          iterator = map_dataset.make_one_shot_iterator
          next_element = iterator.get_next

          result = Graph::Session.run(graph) do |session|
            columns.times.map do
              session.run(next_element)
            end
          end

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

      def test_three_parameters_eager
        Eager::Context.default.as_default do |context|
          dataset = TensorSliceDataset.new(self.three_components)
          map_dataset = MapDataset.new(dataset, self.three_parameters)

          map_dataset.each_with_index do |record, index|
            assert_equal(self.three_components[0][index] ** 2, record[0].value)
            assert_equal(self.three_components[1][index, true] ** 2, record[1].value)
            assert_equal(self.three_components[2][index] ** 2, record[2].value)
          end
        end
      end

      def test_three_parameters_graph
        columns = 7
        Graph::Graph.new.as_default do |graph|
          dataset = TensorSliceDataset.new(self.three_components)
          map_dataset = MapDataset.new(dataset, self.three_parameters)
          iterator = map_dataset.make_one_shot_iterator
          next_element = iterator.get_next

          result = Graph::Session.run(graph) do |session|
            columns.times.map do
              session.run(next_element)
            end
          end

          result.each_with_index do |record, index|
            assert_equal(self.three_components[0][index] ** 2, record[0])
            assert_equal(self.three_components[1][index, true] ** 2, record[1])
            assert_equal(self.three_components[2][index] ** 2, record[2])
          end
        end
      end
    end
  end
end
