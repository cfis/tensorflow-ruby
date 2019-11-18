module Tensorflow
  module Data
    class MapDataset < Dataset
      def initialize(input_dataset, function, other_arguments: [], output_types: nil, output_shapes: nil, use_inter_op_parallelism: nil, preserve_cardinality: nil)
        @output_types = output_types || input_dataset.output_types
        @output_shapes = output_shapes || input_dataset.output_shapes
        variant_tensor = RawOps.map_dataset(input_dataset.variant_tensor, other_arguments,
                                            f: function,
                                            output_types: @output_types,
                                            output_shapes: @output_shapes,
                                            use_inter_op_parallelism: use_inter_op_parallelism,
                                            preserve_cardinality: preserve_cardinality)

        super(variant_tensor)
      end
    end
  end
end
