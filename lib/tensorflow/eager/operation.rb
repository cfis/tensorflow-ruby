module Tensorflow
  module Eager
    class Operation
      attr_reader :context, :op_def, :status

      def initialize(context, op_type, inputs, attrs)
        @context = context
        @op_def = Tensorflow.op_def(op_type)
        @status = Status.new
        @pointer = FFI.TFE_NewOp(context, op_type, self.status)
        name = attrs.delete(:name) || op_type

        setup_attrs(attrs)
        setup_inputs(inputs)
      end

      def to_ptr
        @pointer
      end

      def dtype
        list_ptr = ::FFI::MemoryPointer.new(:int)
        FFI.TFE_OpGetAttrType(self, 'dtype', list_ptr, self.status)
      end

      def setup_attrs(attrs)
        attrs.each do |attr_name, attr_value|
          next unless attr_value

          attr_name = attr_name.to_s
          list_ptr = ::FFI::MemoryPointer.new(:int)
          type = FFI.TFE_OpGetAttrType(self, attr_name, list_ptr, self.status)
          self.status.check
          is_list = Boolean(list_ptr.read_int)

          if is_list
            add_list_attr(type, attr_name, attr_value)
          else
            add_scalar_attr(type, attr_name, attr_value)
          end
        end
      end

      def add_list_attr(type, attr_name, attr_value)
        num_values = attr_value.size

        case type
          when :int
            values = ::FFI::MemoryPointer.new(:int64, num_values)
            values.write_array_of_int64(attr_value)
            FFI.TFE_OpSetAttrIntList(self, attr_name, values, num_values)
          when :float
            values = ::FFI::MemoryPointer.new(:float, num_values)
            values.write_array_of_float(attr_value)
            FFI.TFE_OpSetAttrFloatList(self, attr_name, values, num_values)
          when :shape
            dims_ptrs =
                attr_value.map do |shape|
                  if shape.empty?
                    ptr = ::FFI::MemoryPointer.new(:int64, 1)
                    ptr.write_int64(0)
                  else
                    ptr = ::FFI::MemoryPointer.new(:int64, shape.size)
                    ptr.write_array_of_int64(shape)
                  end
                end
            dims = ::FFI::MemoryPointer.new(:pointer, num_values)
            dims.write_array_of_pointer(dims_ptrs)

            num_dims = ::FFI::MemoryPointer.new(:int, num_values)
            num_dims.write_array_of_int(attr_value.map(&:size))

            FFI.TFE_OpSetAttrShapeList(self, attr_name, dims, num_dims, num_values, self.status)
            self.status.check
          when :type
            values = ::FFI::MemoryPointer.new(:int, num_values)
            types =
                attr_value.map do |v|
                  if v.is_a?(Symbol)
                    FFI::DataType[v]
                  else
                    v
                  end
                end
            values.write_array_of_int(types)
            FFI.TFE_OpSetAttrTypeList(self, attr_name, values, num_values)
          else
            raise "Unknown list type: #{type}"
        end
      end

      def add_scalar_attr(type, attr_name, attr_value)
        case type
          when :string
            FFI.TFE_OpSetAttrString(self, attr_name, attr_value, attr_value.bytesize)
          when :int
            FFI.TFE_OpSetAttrInt(self, attr_name, attr_value)
          when :float
            FFI.TFE_OpSetAttrFloat(self, attr_name, attr_value)
          when :bool
            FFI.TFE_OpSetAttrBool(self, attr_name, attr_value ? 1 : 0)
          when :type
            attr_value = FFI::DataType[attr_value] if attr_value.is_a?(Symbol)
            FFI.TFE_OpSetAttrType(self, attr_name, attr_value)
          when :shape
            ptr = ::FFI::MemoryPointer.new(:int64, attr_value.size)
            ptr.write_array_of_int64(attr_value)
            FFI.TFE_OpSetAttrShape(self, attr_name, ptr, attr_value.size, self.status)
          when :tensor
            attr_value = TensorHandle.from_value(self.context, attr_value)
            FFI.TFE_OpSetAttrTensor(self, attr_name, attr_value.tensor, self.status)
          # when :placeholder
          when :func
            case attr_value
              when Graph::Function
                FFI.TFE_OpSetAttrFunctionName(self, attr_name, attr_value.name, attr_value.name.length)
                #FFI.TFE_OpSetAttrFunction(self, attr_name, attr_value)
              when String
                FFI.TFE_OpSetAttrFunctionName(self, attr_name, attr_value, attr_value.length)
              else
                self.status.set(:tf_invalid_argument, "Invalid function attribute for attribute: #{attr_name}")
            end
          else
            self.status.set(:tf_unknown, "Unsupported attribute type: #{type}")
        end
        self.status.check
      end

      def setup_inputs(inputs)
        inputs.each_with_index do |input, index|
          setup_input(index, input)
        end
      end

      def setup_input(index, value)
        if value.nil?
          self.status.set(:tf_invalid_argument, "Argument is unset. Index: #{index}")
          self.status.check
        end

        arg_def = self.op_def.input_arg[index]

        if !arg_def.number_attr.empty?
          # This input is a homogeneous list
        #  value = TensorHandle.from_value(self.context, value)
          value.each do |a_value|
            a_value = TensorHandle.from_value(self.context, a_value)
            FFI.TFE_OpAddInput(self, a_value, self.status)
            self.status.check
          end
        elsif !arg_def.type_list_attr.empty?
          # This input is a heterogeneous list.
          #value = TensorHandle.from_value(self.context, value)
          #FFI.TFE_OpAddInput(self, value, self.status)
          # values = value.map do |a_value|
          #            Eager::TensorHandle.from_value(self.context, a_value)
          #          end

          value = value.map do |a_value|
            TensorHandle.from_value(self.context, a_value)
          end
          input_ptr = ::FFI::MemoryPointer.new(:pointer, value.size)
          input_ptr.write_array_of_pointer(value)
          FFI.TFE_OpAddInputList(self, input_ptr, value.size, self.status)
        else
          # This should be a single item
          value = if value.is_a?(Array)
                    value = value.map do |a_value|
                      TensorHandle.from_value(self.context, a_value)
                    end
                    value = Tensorflow.pack(value)
                  else
                    TensorHandle.from_value(self.context, value)
                  end
          FFI.TFE_OpAddInput(self, value, self.status)
        end
        self.status.check
      end
    end
  end
end