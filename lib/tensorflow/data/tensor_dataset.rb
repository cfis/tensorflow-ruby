module Tensorflow
  module Data
    class TensorDataset < Dataset
      def initialize(elements)
        @tensors = self.class.to_tensor_array(elements)
        @output_types = @tensors.map(&:dtype)
        @output_shapes = @tensors.map do |tensor|
          tensor.shape
        end

        variant_tensor = RawOps.tensor_dataset(@tensors,
                                               toutput_types: @output_types,
                                               output_shapes: @output_shapes)

        super(variant_tensor)
      end
    end
  end
end
