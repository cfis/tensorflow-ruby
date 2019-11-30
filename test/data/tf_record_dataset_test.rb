 require_relative "../base_test"

module Tensorflow
  module Data
    class TfRecordDatasetTest < BaseTest
      def file_path
        File.expand_path(File.join(__dir__, 'tf_record.0.txt'))
      end

      def test_basic
        self.eager_and_graph do |context|
          dataset = TfRecordDataset.new(file_path)
          assert_equal([:string], dataset.output_types)
          assert_equal([[]], dataset.output_shapes)

          result = self.evaluate(dataset)
          assert_equal(7, result.length)
          assert_equal(["Record 0 of file 0",
                        "Record 1 of file 0",
                        "Record 2 of file 0",
                        "Record 3 of file 0",
                        "Record 4 of file 0",
                        "Record 5 of file 0",
                        "Record 6 of file 0"],
                       result)
        end
      end
    end
  end
end
