require_relative "../base_test"

module Tensorflow
  module Data
    class RepeatDatasetTest < BaseTest
      def test_simple
        components = [1,
                      Numo::NArray[1, 2, 3],
                      37]

        self.eager_and_graph do |context|
          dataset = TensorDataset.new(components).repeat(3)
          result = self.evaluate(dataset)
          assert_equal([[1, [1, 2, 3], 37], [1, [1, 2, 3], 37], [1, [1, 2, 3], 37]], result)
        end
      end
    end
  end
end
