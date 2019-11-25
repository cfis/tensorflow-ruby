module Tensorflow
  module Data
    class Iterator
      attr_reader :output_types, :output_shapes

      def self.from_structure(output_types, output_shapes=[], shared_name: '')
        ReinitializableIterator.new(output_types, output_shapes, shared_name: shared_name)
      end

      def initialize(output_types, output_shapes=[])
        @output_types = output_types
        @output_shapes = output_shapes
      end

      def get_next
        RawOps.iterator_get_next(@iterator, output_types: self.output_types, output_shapes: self.output_shapes)
      end
    end

    class OneShotIterator < Iterator
      def initialize(dataset)
        super(dataset.output_types, dataset.output_shapes)
        create_one_shot_iterator(dataset)
      end

      private

      def create_one_shot_iterator(dataset)
        function = make_dataset_function(dataset)
        ExecutionContext.current.add_function(function)
        @iterator = RawOps.one_shot_iterator(dataset_factory: function, output_types: self.output_types, output_shapes: self.output_shapes)
      end

      def make_dataset_function(dataset)
        function = Graph::Graph.new.as_default do |func_graph|
          optimize = RawOps.optimize_dataset(dataset.variant_tensor, ['noop_elimination'],
                                             output_types: self.output_types, output_shapes: self.output_shapes)
          func_graph.to_function('MakeDataset', nil, nil, [optimize])
        end
      end
    end

    class InitializableIterator < Iterator
      attr_reader :initializer

      def initialize(dataset, shared_name: '')
        super(dataset.output_types, dataset.output_shapes)
        create_initializable_iterator(dataset, shared_name)
      end

      private

      def create_initializable_iterator(dataset, shared_name)
        @iterator = RawOps.iterator_v2(shared_name: shared_name, output_types: self.output_types, output_shapes: self.output_shapes)
        @initializer = RawOps.make_iterator(dataset.variant_tensor, @iterator)
      end
    end

    class ReinitializableIterator < Iterator
      def initialize(output_types, output_shapes, shared_name: '')
        super(output_types, output_shapes)
        create_iterator_from_structure(shared_name)
      end

      def make_initializer(dataset)
        RawOps.make_iterator(dataset.variant_tensor, @iterator)
      end

      private

      def create_iterator_from_structure(shared_name)
        @iterator = RawOps.iterator_v2(shared_name: shared_name, output_types: self.output_types, output_shapes: self.output_shapes)
      end
    end
  end
end