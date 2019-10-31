module Tensorflow
  module TensorMixin
    NUMO_TYPE_MAP = {Numo::Int8   => :int8,
                     Numo::Int16  => :int16,
                     Numo::Int32  => :int32,
                     Numo::Int64  => :int64,
                     Numo::UInt8  => :uint8,
                     Numo::UInt16 => :uint16,
                     Numo::UInt32 => :uint32,
                     Numo::UInt64 => :uint64,
                     Numo::SFloat => :float,
                     Numo::DFloat => :double}

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
  end
end
