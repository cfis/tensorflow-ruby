module Tensorflow
  module Data
    class ShuffleDataset < Dataset
      def initialize(input_dataset, buffer_size)
        @input_dataset = input_dataset # keep reference for memory
        @output_types = input_dataset.output_types
        @output_shapes = input_dataset.output_shapes

        variant_tensor = RawOps.shuffle_dataset(input_dataset,
                                                buffer_size,
                                                seed,
                                                seed2,
                                                output_types: @output_types,
                                                output_shapes: @output_shapes)
        super(variant_tensor)
      end
    end
  end
end
