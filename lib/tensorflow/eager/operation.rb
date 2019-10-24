module Tensorflow
  module Eager
    class Operation
      attr_reader :context, :op_def

      def initialize(context)
        @context = context
      end

      def execute(op_name, inputs = [], **attrs)
        @op_def = Graph::Operation.op_def(op_name)

        status = Status.new
        op = FFI.TFE_NewOp(context, op_name, status)

        attrs.each do |attr_name, attr_value|
          next unless attr_value

          attr_name = attr_name.to_s
          is_list = ::FFI::MemoryPointer.new(:int)
          type = FFI.TFE_OpGetAttrType(op, attr_name, is_list, status)
          status.check

          if is_list.read_int == 1
            add_list_attr(op, status, type, attr_name, attr_value)
          else
            add_scalar_attr(op, status, type, attr_name, attr_value)
          end
        end

        inputs.each_with_index do |input, i|
          add_input(op, status, input, i)
        end

        read_result(op, status)
        # ensure
        #   FFI.TF_DeleteStatus(status) if status
        #   FFI.TFE_DeleteOp(op) if op
      end

      def add_list_attr(op, status, type, attr_name, attr_value)
        num_values = attr_value.size

        case FFI::AttrType[type]
          when :int
            values = ::FFI::MemoryPointer.new(:int64, num_values)
            values.write_array_of_int64(attr_value)
            FFI.TFE_OpSetAttrIntList(op, attr_name, values, num_values)
          when :float
            values = ::FFI::MemoryPointer.new(:float, num_values)
            values.write_array_of_float(attr_value)
            FFI.TFE_OpSetAttrFloatList(op, attr_name, values, num_values)
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

            FFI.TFE_OpSetAttrShapeList(op, attr_name, dims, num_dims, num_values, status)
            status.check
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
            FFI.TFE_OpSetAttrTypeList(op, attr_name, values, num_values)
          else
            raise "Unknown list type: #{FFI::AttrType[type]}"
        end
      end

      def add_scalar_attr(op, status, type, attr_name, attr_value)
        case FFI::AttrType[type]
          when :string
            FFI.TFE_OpSetAttrString(op, attr_name, attr_value, attr_value.bytesize)
          when :int
            FFI.TFE_OpSetAttrInt(op, attr_name, attr_value)
          when :float
            FFI.TFE_OpSetAttrFloat(op, attr_name, attr_value)
          when :bool
            FFI.TFE_OpSetAttrBool(op, attr_name, attr_value ? 1 : 0)
          when :type
            attr_value = FFI::DataType[attr_value] if attr_value.is_a?(Symbol)
            FFI.TFE_OpSetAttrType(op, attr_name, attr_value)
          when :shape
            ptr = ::FFI::MemoryPointer.new(:int64, attr_value.size)
            ptr.write_array_of_int64(attr_value)
            FFI.TFE_OpSetAttrShape(op, attr_name, ptr, attr_value.size, status)
          when :tensor
            FFI.TFE_OpSetAttrTensor(op, attr_name, attr_value.tensor_pointer, status)
          # when :placeholder
          when :func
            case attr_value
              when Graph::Function
                FFI.TFE_OpSetAttrFunctionName(op, attr_name, attr_value.name, attr_value.name.length)
                #FFI.TFE_OpSetAttrFunction(op, attr_name, attr_value)
              when String
                FFI.TFE_OpSetAttrFunctionName(op, attr_name, attr_value, attr_value.length)
              else
                status.set(:tf_invalid_argument, "Invalid function attribute for attribute: #{attr_name}")
            end
          else
            status.set(:tf_unknown, "Unsupported attribute type: #{FFI::AttrType[type]}")
        end
        status.check
      end

      def add_input(op, status, input, i)
        raise "Missing argument" if input.nil?

        arg_def = self.op_def.input_arg[i]

        if !arg_def.number_attr.empty?
          # This input is a homogeneous list
          input.each do |a_input|
            a_input = Eager.convert_to_tensor_handle(a_input)
            FFI.TFE_OpAddInput(op, a_input, status)
            status.check
          end
        elsif !arg_def.type_list_attr.empty?
          # This input is a heterogeneous list.
          inputs = input.map do |a_input|
                     Eager.convert_to_tensor_handle(a_input)
                   end

          input_ptr = ::FFI::MemoryPointer.new(:pointer, inputs.size)
          input_ptr.write_array_of_pointer(inputs)
          FFI.TFE_OpAddInputList(op, input_ptr, inputs.size, status)
        else
          # This input is a single item
          input = Eager.convert_to_tensor_handle(input)
          FFI.TFE_OpAddInput(op, input, status)
        end
        status.check
      end

      def read_result(op, status)
        # TODO decide how many retvals to allocate
        retvals = ::FFI::MemoryPointer.new(:pointer, 10)
        num_retvals = ::FFI::MemoryPointer.new(:int)
        num_retvals.write_int(retvals.size)
        FFI.TFE_Execute(op, retvals, num_retvals, status)
        status.check

        n = num_retvals.read_int
        if n > 0
          handles = retvals.read_array_of_pointer(n).map do |handle|
                      TensorHandle.new(handle)
                    end

          # TODO handle case where n = 1 and still want an array for retvals
          n == 1 ? handles.first : handles
        end
      end
    end
  end
end
