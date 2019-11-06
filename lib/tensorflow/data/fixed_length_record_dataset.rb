module Tensorflow
  module Data
    class FixedLengthRecordDataset < Dataset
      def initialize(filenames, record_bytes, header_bytes: 0, footer_bytes: 0, buffer_size: DEFAULT_READER_BUFFER_SIZE_BYTES, compression_type: '')
        @output_types = [:string]
        @output_shapes = [[]]
        variant_tensor = RawOps.fixed_length_record_dataset_v2(filenames,
                                                               header_bytes,
                                                               record_bytes,
                                                               footer_bytes,
                                                               buffer_size,
                                                               compression_type)

        super(variant_tensor)
      end
    end
  end
end
