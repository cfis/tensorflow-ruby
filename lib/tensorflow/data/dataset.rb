module Tensorflow
  module Data
    class Dataset
      # Copied from Python code
      DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB

      include Enumerable

      # TODO remove
      attr_reader :output_types, :output_shapes, :variant_tensor

      def self.to_tensor_array(values)
        case values
          when Numo::NArray
            [Tensor.new(values)]
          when Tensor
            [values]
          else
            values.to_a.map do |v|
              if v.is_a?(Tensor)
                v
              else
                Tensor.new(v)
              end
            end
        end
      end

      def self.from_tensor_slices(tensors)
        TensorSliceDataset.new(tensors)
      end

      def initialize(variant_tensor)
        @variant_tensor = variant_tensor
      end

      def to_ptr
        @variant_tensor.to_ptr
      end

      def with_options(options)

      end
      def batch(batch_size, drop_remainder: false)
        BatchDataset.new(self, batch_size, drop_remainder)
      end

      def shuffle(buffer_size)
        ShuffleDataset.new(self, buffer_size)
      end

      def make_one_shot_iterator
        Iterator.new(self)
      end

      def each
        iterator, deleter = RawOps.anonymous_iterator_v2(output_types: @output_types, output_shapes: @output_shapes)
        RawOps.make_iterator(@variant_tensor, iterator)
        begin
          loop do
            values = RawOps.iterator_get_next_sync(iterator, output_types: @output_types, output_shapes: @output_shapes)
            yield values
          end
        rescue ::TensorflowError => e
          # iterate until end of sequence error
          raise e unless e.message == "End of sequence"
        end
      ensure
        RawOps.delete_iterator(iterator, deleter) if iterator
      end

      # !!! DEBUG method. You don't want to use this method it because it iterates over
      # the entire dataset and reads it into a ruby array in memory
      def data
        self.map do |slice|
          if slice.is_a?(Array)
            slice.map do |tensor|
              tensor.value
            end
          else
            slice.value
          end
        end
      end

      # def map(&block)
      #   MapDataset.new(self, nil, block)
      # end

      def repeat(count)
        RepeatDataset.new(self, 3)
      end
    end
  end
end
