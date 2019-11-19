module Tensorflow
  module Data
    class ShuffleDataset < Dataset
      def initialize(input_dataset, buffer_size)
        @input_dataset = input_dataset
        @output_types = input_dataset.output_types
        @output_shapes = input_dataset.output_shapes

        buffer_size = Tensor.new(buffer_size, dtype: :int64)
        seed = Tensor.new(::Random.rand(100_000_000), dtype: :int64)
        seed2 = Tensor.new(::Random.rand(100_000_000), dtype: :int64)

        variant_tensor = RawOps.shuffle_dataset(input_dataset.variant_tensor,
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
