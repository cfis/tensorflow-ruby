module Tensorflow
  module Data
    class Iterator
      attr_reader :dataset

      def initialize(dataset)
        @dataset = dataset
        @iterator = one_shot_iterator
      end

      def get_next
        RawOps.iterator_get_next(@iterator, output_types: self.dataset.output_types, output_shapes: self.dataset.output_shapes)
      end

      private

      def make_dataset_function
        function = Graph::Graph.new.as_default do |func_graph|
          optimize = RawOps.optimize_dataset(dataset.variant_tensor, ['noop_elimination'],
                                             output_types: self.dataset.output_types, output_shapes: self.dataset.output_shapes)
          func_graph.to_function('MakeDataset', nil, nil, [optimize])
        end
      end

      def one_shot_iterator
        function = make_dataset_function
        ExecutionContext.current.add_function(function)
        RawOps.one_shot_iterator(dataset_factory: function, output_types: dataset.output_types, output_shapes: dataset.output_shapes)
      end
    end
  end
end

