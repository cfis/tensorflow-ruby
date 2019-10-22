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
          klass = Utils::NUMO_TYPE_MAP[dtype]
          raise "Unknown type: #{dtype}" unless klass
          klass.cast(value)
      end
    end

    def +(other)
      Math.add(self, other)
    end

    def -(other)
      Math.subtract(self, other)
    end

    def *(other)
      Math.multiply(self, other)
    end

    def /(other)
      Math.divide(self, other)
    end

    def %(other)
      Math.floormod(self, other)
    end

    def -@
      Math.negative(self)
    end
  end
end
