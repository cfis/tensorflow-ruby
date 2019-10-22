module Tensorflow
  module Graph
    class Graph
      def self.finalize(pointer)
        proc do
          FFI::TF_DeleteGraph(pointer)
        end
      end

      def initialize
        @number_of_defaults_created = Hash.new(0)
        @pointer = FFI.TF_NewGraph()
        ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
      end

      def to_ptr
        @pointer
      end

      def op_def(name)
        buffer = FFI::Buffer.new
        Status.check do |status|
          FFI.TF_GraphGetOpDef(self, name, buffer, status)
        end
        string = buffer[:data].read_string(buffer[:length])
        ops = OpDef.decode(string)
      end

      def constant(tensor, name)
        op_desc = OperationDescription.new(self, 'Const', name)
        op_desc.attr('value').tensor = tensor
        op_desc.attr('dtype').tensor = tensor.dtype
        op_desc.save
      end

      def operations
        result = Array.new
        position = 0
        position_ptr = ::FFI::MemoryPointer.new(:size_t, 1, true)
        position_ptr.write_int(position)
        while (ptr = FFI.TF_GraphNextOperation(self, position_ptr))
          break if ptr.null?
          result << Operation.new(ptr)
          position_ptr.write_int(position += 1)
        end

        result
      end

      def operation(name)
        ptr = FFI.TF_GraphOperationByName(self, name)
        ptr.null? ? nil : Operation.new(ptr)
      end

      def tensor_num_dims(operation)
        output = FFI::Output.new
        output[:oper] = operation
        output[:index] = 0
        Status.check do |status|
          FFI.TF_GraphGetTensorNumDims(self, output, status)
        end
      end

      def tensor_get_shape(operation)
        length = self.tensor_num_dims(operation)
        return [-1] if length == -1
        ptr = ::FFI::MemoryPointer.new(:int64, length)
        output = FFI::Output.new
        output[:oper] = operation
        output[:index] = 0
        Status.check do |status|
          FFI.TF_GraphGetTensorShape(self, output, ptr, length, status)
        end
        ptr.read_array_of_int64(length)
      end

      def tensor_set_shape(operation, shape)
        ptr = ::FFI::MemoryPointer.new(:int64, shape.length)
        ptr.write_array_of_int64(shape)
        output = FFI::Output.new
        output[:oper] = operation
        output[:index] = 0
        Status.check do |status|
          FFI.TF_GraphSetTensorShape(self, output, ptr, shape.length, status)
        end
      end

      def to_function(name, operators, inputs, outputs, output_names)
        inputs_ptr = inputs ? FFI::Output.pointer_array(inputs) : nil
        outputs_ptr = outputs ? FFI::Output.pointer_array(outputs) : nil

        output_names_ptr = ::FFI::MemoryPointer.new(:pointer, output_names.length, true)
        output_names.each do |output_name|
          output_name_ptr = ::FFI::MemoryPointer.from_string(output_name)
          output_names_ptr.write_pointer(output_name_ptr)
        end

        append_hash_to_fn_name = 0
        options = nil
        description = nil

        func = Status.check do |status|
          FFI.TF_GraphToFunction(self, name, append_hash_to_fn_name,
                                 operators ? operators.length : -1, operators,
                                 inputs ? inputs.length : 0, inputs_ptr,
                                 outputs ? outputs.length: 0, outputs_ptr,
                                 output_names_ptr,
                                 options, description, status)
        end
        Function.new(func)
      end

      def placeholder(name, dtype=:int32)
        op_desc = OperationDescription.new(self, 'Placeholder', name)
        op_desc.attr('dtype').dtype = dtype
        op_desc.save
      end

      def constant(value, name=nil)
        tensor = value.is_a?(Tensor) ? value : Tensor.new(value)
        op_desc = OperationDescription.new(self, 'Const', name)
        op_desc.attr('value').tensor = tensor
        op_desc.attr('dtype').dtype = tensor.dtype
        op_desc.save
      end

      # write_to writes out a serialized representation of graph in binary wire format.
      # This graph defination can be written to file using write_file function and then
      # converted to a readable form using converter.py file in the gem.
      def write_to
        buffer = Tensorflow::TF_NewBuffer()
        status = Tensorflow::Status.new
        Tensorflow::TF_GraphToGraphDef(c, buffer, status.c)
        Tensorflow.buffer_write(buffer)
      end

      # write_file writes out a serialized representation of graph to a file.
      def write_file(filename)
        File.open(filename, 'w') { |file| file.write(write_to) }
      end

    end
  end
end