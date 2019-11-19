module Tensorflow
  module Eager
    class Operation
      attr_reader :context, :guessed_dtype, :op_def, :status

      def initialize(context, op_type, inputs, attrs)
        @context = context
        @op_def = case op_type
                    when Graph::Function
                      op_type.function_def.signature
                    else
                      Tensorflow.op_def(op_type)
                  end
        raise(TensorflowError, "Invalid op type: #{op_type}") unless @op_def

        @status = Status.new
        @pointer = FFI.TFE_NewOp(context, self.op_def.name, self.status)
        name = attrs.delete(:name) || op_type

        inputs = Array(inputs)
        @guessed_dtype = figure_dtype(attrs, inputs)

        setup_inputs(inputs, attrs)
        setup_attrs(attrs)
      end

      def to_ptr
        @pointer
      end

      def dtype
        list_ptr = ::FFI::MemoryPointer.new(:int)
        FFI.TFE_OpGetAttrType(self, 'dtype', list_ptr, self.status)
      end

      def figure_dtype(attrs, inputs)
        attr_def = self.op_def.attr.detect do |attr_def|
          attr_def.type == 'type'
        end

        result = attr_def ? attrs[attr_def.name.to_sym] : nil
        unless result
          inputs.each do |input|
            case input
              when Operation
                return input.output_types.first
              when Variable
                return input.dtype
            end
          end
        end
        result
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
            dims_pointer = ::FFI::MemoryPointer.new(:pointer, num_values)
            num_dims_pointer = ::FFI::MemoryPointer.new(:int32, num_values)
            attr_value.each_with_index do |shape, i|
              dim_pointer = ::FFI::MemoryPointer.new(:int64, shape.length)
              dim_pointer.write_array_of_int64(shape)
              dims_pointer.put_pointer(i * ::FFI.type_size(:pointer), dim_pointer)
              num_dims_pointer.put_int32(i * ::FFI.type_size(:int32), shape.length)
            end
            FFI.TFE_OpSetAttrShapeList(self, attr_name, dims_pointer, num_dims_pointer, num_values, self.status)
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

      def setup_inputs(inputs, attrs)
        inputs.each_with_index do |input, index|
          setup_input(index, input, attrs)
        end
      end

      def check_input(arg_def, input, dtype)
        case input
          when Variable
            arg_def.type == :DT_RESOURCE ? input.handle : input.value_handle
          else
            TensorHandle.from_value(self.context, input, dtype: dtype)
        end
      end

      def setup_input(index, value, attrs)
        if value.nil?
          self.status.set(:tf_invalid_argument, "Argument is unset. Index: #{index}")
          self.status.check
        end

        arg_def = self.op_def.input_arg[index]
        dtype = attrs[arg_def.type_attr.to_sym]

        # Value can be an operation with multiple outputs. For example calling PACK with an input operation of SPLIT
        checked_value = if (!arg_def.number_attr.empty? || !arg_def.type_list_attr.empty?)  && value.is_a?(Array)
                          value.map do |sub_value|
                            self.check_input(arg_def, sub_value, dtype)
                          end
                        else
                          self.check_input(arg_def, value, dtype)
                        end

        if !arg_def.type_list_attr.empty?
          # This input is a heterogeneous list
          self.add_input_list(checked_value)
        elsif !arg_def.number_attr.empty? && !arg_def.type_attr.empty?
          # This input is a homogeneous list
          self.add_input_list(checked_value)
        elsif !arg_def.number_attr.empty?
          # This is a list but we have to set it up one input at a time
          checked_value.each do |sub_checked_value|
            self.add_input(sub_checked_value)
          end
        else
          # This input is a single item
          self.add_input(checked_value)
        end
      end

      def add_input(value)
        # Check to see if the operation has multiple outputs, and if it does, we need to pack them together
        # to fit into one input
        if value.is_a?(Array) && value.length > 1
          packed = Tensorflow.pack(value)
          FFI.TFE_OpAddInput(self, packed, self.status)
        else
          FFI.TFE_OpAddInput(self, value, self.status)
        end
        self.status.check
      end

      def add_input_list(values)
        input_ptr = ::FFI::MemoryPointer.new(:pointer, values.length)
        input_ptr.write_array_of_pointer(values)
        FFI.TFE_OpAddInputList(self, input_ptr, values.length, self.status)
        self.status.check
      end
    end
  end
end