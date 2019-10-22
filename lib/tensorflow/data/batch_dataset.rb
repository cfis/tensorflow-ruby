module Tensorflow
  module Data
    class BatchDataset < Dataset
      def initialize(input_dataset, batch_size, drop_remainder)
        @input_dataset = input_dataset # keep reference for memory
        @output_types = input_dataset.output_types
        @output_shapes = input_dataset.output_shapes.map { |s| [batch_size] + s }

        variant_tensor = RawOps.batch_dataset_v2(
          input_dataset: input_dataset,
          batch_size: Eager.convert_to_tensor_handle(batch_size, dtype: :int64),
          drop_remainder: drop_remainder,
          output_types: @output_types,
          output_shapes: @output_shapes
        )
        super(variant_tensor)
      end
    end
  end
end
