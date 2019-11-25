module Tensorflow
  module Data
    class FixedLengthRecordDataset < Dataset
      def initialize(filenames, record_bytes, header_bytes: 0, footer_bytes: 0,
                     buffer_size: DEFAULT_READER_BUFFER_SIZE_BYTES, compression_type: '', num_parallel_reads: 0)
        @output_types = [:string]
        @output_shapes = [[]]

        record_bytes_tensor = Tensor.new(record_bytes, dtype: :int64)
        header_bytes_tensor = Tensor.new(header_bytes, dtype: :int64)
        footer_bytes_tensor = Tensor.new(footer_bytes, dtype: :int64)
        buffer_size_tensor = Tensor.new(buffer_size, dtype: :int64)

        variant_tensor = RawOps.fixed_length_record_dataset_v2(filenames,
                                                               header_bytes_tensor,
                                                               record_bytes_tensor,
                                                               footer_bytes_tensor,
                                                               buffer_size_tensor,
                                                               compression_type)

        super(variant_tensor)
      end


    end
  end
end
