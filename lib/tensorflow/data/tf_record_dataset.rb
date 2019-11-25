module Tensorflow
  module Data
    class TfRecordDataset < Dataset
      DEFAULT_BUFFER_SIZE = 256 * 1_048_576  # 256 MB

      def initialize(filenames, compression_type='', buffer_size=DEFAULT_BUFFER_SIZE)
        filenames = Array(filenames)
        @output_types = [:string]
        @output_shapes = [[]]

        buffer_size = Tensor.new(buffer_size, dtype: :int64) if buffer_size
        variant_tensor = RawOps.tf_record_dataset(filenames, compression_type, buffer_size)

        super(variant_tensor)
      end
    end
  end
end
