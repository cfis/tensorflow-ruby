module Tensorflow
  module TensorMixin
    def shape
      @shape ||= begin
        status = Status.new
        shape = []
        if self
          num_dims.times do |i|
            shape << dim(i)
            status.check
          end
        end
        shape
      end
    end

    def numo
      case dtype
        when NilClass
          nil
        when :variant
          :variant
        when :string
          :string
        else
          klass = Utils::DTYPE_TO_NUMO_TYPE_MAP[dtype]
          raise "Unknown type: #{dtype}" unless klass
          klass.cast(value)
      end
    end
  end
end
