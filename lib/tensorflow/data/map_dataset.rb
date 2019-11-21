module Tensorflow
  module Data
    class MapDataset < Dataset
      def initialize(input_dataset, function, other_arguments: [])
        @output_types = function.output_types
        @output_shapes = function.output_shapes

        variant_tensor = RawOps.map_dataset(input_dataset.variant_tensor, other_arguments,
                                            f: function,
                                            output_types: @output_types,
                                            output_shapes: @output_shapes)

        super(variant_tensor)
      end
    end
  end
end
