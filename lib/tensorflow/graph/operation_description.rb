module Tensorflow
  module Graph
    class OperationDescription
      attr_reader :op_def

      def initialize(graph, op_name, name)
        @op_def = self.check_op_def(graph, op_name)
        unless @op_def
          raise(::TensorflowError, "Unknown operation: #{op_name}")
        end
        @pointer = FFI.TF_NewOperation(graph, op_name, name)
      end

      def check_op_def(graph, op_name)
        buffer_ptr = FFI.TF_NewBuffer
        Status.check do |status|
          FFI.TF_GraphGetOpDef(graph, op_name, buffer_ptr, status)
        end
        buffer = FFI::Buffer.new(buffer_ptr)
        string = buffer[:data].read_string(buffer[:length])
        OpDef.decode(string)
      ensure
        FFI.TF_DeleteBuffer(buffer)
      end

      def to_ptr
        @pointer
      end

      def device=(value)
        FFI.TF_SetDevice(self, value)
      end

      def attr(attr_name)
        attr_def = self.op_def.attr.detect do |attr_def|
          attr_def.name == attr_name
        end
        unless attr_def
          raise(::TensorflowError, "Unknown attribute: #{attr_name}")
        end

        OperationDescriptionAttr.new(self, attr_def)
      end

      def add_input(operation)
        input = FFI::Output.new
        input[:oper] = operation
        input[:index] = 0
        FFI.TF_AddInput(self, input)
      end

      def add_inputs(*operations)
        operations = operations.flatten(1)

        operations_ptr = FFI::Output.pointer_array(operations)
        FFI.TF_AddInputList(self, operations_ptr, operations.length)
      end

      def save
        Status.check do |status|
          ptr = FFI.TF_FinishOperation(self, status)
          Operation.new(ptr)
        end
      end
    end

    class OperationDescriptionAttr
      attr_reader :attr_def, :operation_description
      def initialize(operation_description, attr_def)
        @operation_description = operation_description
        @attr_def = attr_def
      end

      def dtype=(value)
        @dtype = value
        FFI.TF_SetAttrType(self.operation_description, self.attr_def.name, value)
      end

      def shape=(value)
        @shape = value
        pointer = ::FFI::MemoryPointer.new(:int64, value.length)
        pointer.write_array_of_int64(value)
        FFI.TF_SetAttrShape(self.operation_description, self.attr_def.name, pointer, value.length)
      end

      def tensor=(value)
        Status.check do |status|
          FFI.TF_SetAttrTensor(self.operation_description, self.attr_def.name, value, status)
        end
      end

      def value=(value)
        @value = value
        if value.is_a?(Array)
          self.add_list_attr(value)
        else
          self.add_scalar_attr(value)
        end
      end

      def add_list_attr(values)
        num_values = attr_value.size

        case FFI::AttrType[type]
          when :string
            values = ::FFI::MemoryPointer.new(:int64, num_values)
          when :int
            values = ::FFI::MemoryPointer.new(:int64, num_values)
            values.write_array_of_int64(attr_value)
            FFI.TFE_OpSetAttrIntList(op, attr_name, values, num_values)
          when :float
            values = ::FFI::MemoryPointer.new(:float, num_values)
            values.write_array_of_float(attr_value)
            FFI.TFE_OpSetAttrFloatList(op, attr_name, values, num_values)
          else
            raise "Unknown list type: #{FFI::AttrType[type]}"
        end
      end

      def add_scalar_attr(value)
        case FFI::AttrType[type]
          when :string
            FFI.TF_SetAttrString(self, self.attr_def.name, value, value.bytesize)
          when :int
            FFI.TF_SetAttrInt(self, self.attr_def.name, value)
          when :float
            FFI.TF_SetAttrFloat(self, self.attr_def.name, value)
          when :bool
            FFI.TF_SetAttrBool(self, self.attr_def.name, value ? 1 : 0)
          else
            raise "Unknown type: #{FFI::AttrType[type]}"
        end
      end
    end
  end
end