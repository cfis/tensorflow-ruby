module Tensorflow
  module Data
    class BatchDataset < Dataset
      def initialize(input_dataset, batch_size, drop_remainder)
        @input_dataset = input_dataset
        @output_types = input_dataset.output_types
        @output_shapes = input_dataset.output_shapes.map do |shape|
          shape.unshift(-1)
        end

        batch_size = Tensor.new(batch_size, dtype: :int64)
        variant_tensor = RawOps.batch_dataset_v2(input_dataset.variant_tensor, batch_size, drop_remainder,
                                                 output_types: @output_types, output_shapes: @output_shapes)
        super(variant_tensor)
      end
    end
  end
end
