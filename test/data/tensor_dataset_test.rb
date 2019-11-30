 require_relative "../base_test"

module Tensorflow
  module Data
    class TensorDatasetTest < BaseTest
      def test_simple
        self.eager_and_graph do |context|
          components = [1,
                        Numo::NArray[1, 2, 3],
                        37]

          dataset = TensorDataset.new(components)

          assert_equal([:int32, :int32 , :int32], dataset.output_types)
          assert_equal([[], [3], []], dataset.output_shapes)

          result = self.evaluate(dataset)
          assert_equal([[1, [1, 2, 3], 37]], result)
        end
      end
    end
  end
end
