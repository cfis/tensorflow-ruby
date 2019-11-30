 require_relative "../base_test"

module Tensorflow
  module Data
    class ShuffleDatasetTest < BaseTest
      def test_simple
        self.eager_and_graph do |context|
          components = [Numo::NArray[1, 2, 3, 4],
                        Numo::NArray[5, 6, 7, 8],
                        Numo::NArray[9.0, 10.0, 11.0, 12.0]]
          dataset = TensorSliceDataset.new(components).shuffle(100)
          result = self.evaluate(dataset)

          assert_equal([[1, 5, 9.0],
                        [2, 6, 10.0],
                        [3, 7, 11.0],
                        [4, 8, 12.0]], result.sort)
        end
      end
    end
  end
end
