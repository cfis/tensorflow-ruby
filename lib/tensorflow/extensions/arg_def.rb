module Tensorflow
  class OpDef
    class ArgDef
      def dtype
        case self.type
          when :DT_INVALID
            nil
          when :DT_FLOAT
            :float
          when :DT_DOUBLE
            :double
          when :DT_INT32
            :int32
          when :DT_UINT8
            :uint8
          when :DT_INT16
            :int16
          when :DT_INT8
            :int8
          when :DT_STRING
            :string
          when :DT_COMPLEX64
            :complex64
          when :DT_INT64
            :int64
          when :DT_BOOL
            :bool
          when :DT_QINT8
            :qint8
          when :DT_QUINT8
            :quint8
          when :DT_QINT32
            :qint32
          when :DT_BFLOAT16
            :bfloat16
          when :DT_QINT16
            :qint16
          when :DT_QUINT16
            :quint16
          when :DT_UINT16
            :uint16
          when :DT_COMPLEX128
            :complex128
          when :DT_HALF
            :half
          when :DT_RESOURCE
            :resource
          when :DT_VARIANT
            :variant
          when :DT_UINT32
            :uint32
          when :DT_UINT64
            :uint64
        end
      end
    end
  end
end
