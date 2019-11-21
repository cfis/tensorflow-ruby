 require_relative "../base_test"

module Tensorflow
  module Data
    class BatchDatasetTest < BaseTest
      extend Decorator

      def components(columns)
        [Numo::Int32.new(columns).seq,
         Numo::NArray[[1, 2, 3]] * Numo::Int32.new(columns).seq[true, :new],
         Numo::NArray[37.0] * Numo::Int32.new(columns).seq]
      end

      def test_one_batch
        batch_size = 2
        self.eager_and_graph do |context|
          dataset = TensorSliceDataset.new(components(batch_size)).batch(batch_size)
          assert_equal([:int32, :int32, :double], dataset.output_types)
          assert_equal([[-1], [-1, 3], [-1]], dataset.output_shapes)

          result = self.result(context, dataset)
          assert_equal(1, result.length)
          assert_equal([[0, 1],
                        [[0, 0, 0], [1, 2, 3]],
                        [0.0, 37.0]], result[0])
        end
      end

      def test_two_batch
        batch_size = 2
        self.eager_and_graph do |context|
          dataset = TensorSliceDataset.new(components(batch_size * 2)).batch(batch_size)
          assert_equal([:int32, :int32, :double], dataset.output_types)
          assert_equal([[-1], [-1, 3], [-1]], dataset.output_shapes)

          result = self.result(context, dataset)
          assert_equal(2, result.length)

          assert_equal([[0, 1],
                        [[0, 0, 0], [1, 2, 3]],
                        [0.0, 37.0]], result[0])

          assert_equal([[2, 3],
                        [[2, 4, 6], [3, 6, 9]],
                        [74.0, 111.0]], result[1])
        end
      end

      def test_partial_batch
        batch_size = 2
        self.eager_and_graph do |context|
          dataset = TensorSliceDataset.new(components(batch_size +1)).batch(batch_size)
          assert_equal([:int32, :int32, :double], dataset.output_types)
          assert_equal([[-1], [-1, 3], [-1]], dataset.output_shapes)

          result = self.result(context, dataset)
          assert_equal(2, result.length)

          assert_equal([[0, 1],
                        [[0, 0, 0], [1, 2, 3]],
                        [0.0, 37.0]], result[0])

          assert_equal([[2],
                        [[2, 4, 6]],
                        [74.0]], result[1])
        end
      end

      def test_partial_batch_drop
        batch_size = 2
        self.eager_and_graph do |context|
          dataset = TensorSliceDataset.new(components(batch_size +1)).batch(batch_size, drop_remainder: true)
          assert_equal([:int32, :int32, :double], dataset.output_types)
          assert_equal([[-1], [-1, 3], [-1]], dataset.output_shapes)

          result = self.result(context, dataset)
          assert_equal(1, result.length)

          assert_equal([[0, 1],
                        [[0, 0, 0], [1, 2, 3]],
                        [0.0, 37.0]], result[0])
        end
      end
    end
  end
end
