 require_relative "../base_test"

module Tensorflow
  module Data
    class TensorSliceDatasetTest < BaseTest
      def test_array
        self.eager_and_graph do |context|
          components = Numo::NArray[[1], [2], [3]]
          dataset = TensorSliceDataset.new(components)
          
          result = self.result(context, dataset)
          assert_equal([[1], [2], [3]], result)
        end
      end

      def test_narray_eager
        self.eager_and_graph do |context|
          components = [Numo::NArray[[1], [2], [3], [4]].tile(20),
                        Numo::NArray[[12], [13], [14], [15]].tile(22),
                        Numo::NArray[37.0, 38.0, 39.0, 40.0]]

          dataset = TensorSliceDataset.new(components)
          result = self.result(context, dataset)

          result.each_with_index do |slice, i|
            assert_equal(components[0].to_a[i], slice[0])
            assert_equal(components[1].to_a[i], slice[1])
            assert_equal(components[2].to_a[i], slice[2])
          end

          assert_equal([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                         37.0],
                        [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
                         38.0],
                        [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                         [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
                         39.0],
                        [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                         [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                         40.0]], result)
        end
      end
    end
  end
end
