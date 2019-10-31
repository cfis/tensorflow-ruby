module Tensorflow
  module Graph
    class OperationDescription
      attr_reader :graph, :name, :op_def

      def initialize(graph, op_type, inputs, attrs)
        @graph = graph
        @op_def = self.get_op_def(op_type)
        name = attrs.delete(:name) || op_type
        @name = self.graph.scoped_name(name)
        @pointer = FFI.TF_NewOperation(graph, op_type, @name)
        setup_inputs(Array(inputs))
        setup_attrs(**attrs)
      end

      def get_op_def(op_type)
        buffer_ptr = FFI.TF_NewBuffer
        Status.check do |status|
          FFI.TF_GraphGetOpDef(self.graph, op_type, buffer_ptr, status)
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

      def save
        Status.check do |status|
          ptr = FFI.TF_FinishOperation(self, status)
          Operation.new(self.graph, ptr)
        end
      end

      def device=(value)
        FFI.TF_SetDevice(self, value)
      end

      def check_input(arg_def, input)
        case input
          when Operation
            input
          when Array
            input_name = "#{self.name}/#{arg_def.name}"
            self.graph.constant(input, name: input_name)
          else
            input_name = "#{self.name}/#{arg_def.name}"
            self.graph.constant(input, name: input_name)
        end
      end

      def setup_inputs(inputs)
        inputs.each_with_index do |input, index|
          self.setup_input(index, input)
        end
      end

      def setup_input(index, value)
        arg_def = self.op_def.input_arg[index]
        value = self.check_input(arg_def, value)

        if !arg_def.number_attr.empty?
          # This input is a homogeneous list
          self.add_input_list(value) #addn operation required a list, but something else I don't remember wanted each input separate?
          # value.each do |a_value|
          #   self.add_input(a_value)
          # end
        elsif !arg_def.type_list_attr.empty?
          self.add_input_list(value)
        else
          # This input is a single item
          self.add_input(value)
        end
      end

      def add_input(operation)
        input = FFI::Output.new
        input[:oper] = operation
        input[:index] = 0
        FFI.TF_AddInput(self, input)
      end

      def add_input_list(operations)
        operations_ptr = FFI::Output.pointer_array(operations)
        FFI.TF_AddInputList(self, operations_ptr, operations.length)
      end

      def setup_attrs(**attrs)
        attrs.each do |attr_name, attr_value|
          self.setup_attr(attr_name, attr_value)
        end
      end

      def setup_attr(name, value)
        attr_def = self.op_def.attr.detect do |attr_def|
          name.to_s == attr_def.name
        end
        unless attr_def
          raise(::TensorflowError, "Unknown attribute: #{name}")
        end

        case attr_def.type
          when 'bool'
            FFI.TF_SetAttrBool(self, attr_def.name, value ? 1 : 0)
          when 'int'
            FFI.TF_SetAttrInt(self, attr_def.name, value)
          when 'float'
            FFI.TF_SetAttrFloat(self, attr_def.name, value)
          when 'func'
            FFI.TF_SetAttrFuncName(self, attr_def.name, value, value.length)
          when 'shape'
            pointer = ::FFI::MemoryPointer.new(:int64, value.length)
            pointer.write_array_of_int64(value)
            FFI.TF_SetAttrShape(self, attr_def.name, pointer, value.length)
          when 'string'
            FFI.TF_SetAttrString(self, attr_def.name, value, value.length)
          when 'tensor'
            Status.check do |status|
              FFI.TF_SetAttrTensor(self, attr_def.name, value, status)
            end
          when 'type'
            FFI.TF_SetAttrType(self, attr_def.name, value)
          else
            raise(TensorflowError, "Unsupported attribute. #{self.op_def.name} - #{attr_def.name}")
        end
      end
    end
  end
end