module Tensorflow
  module Graph
    class Graph
      attr_reader :name_scope
      extend Forwardable
      def_delegators :@name_scope, :name_scope, :scoped_name

      def self.finalize(pointer)
        proc do
          FFI::TF_DeleteGraph(pointer)
        end
      end

      def initialize
        @name_scope = NameScope.new
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

      def create_operation(op_name, name=nil)
        op_desc = OperationDescription.new(self, op_name, name)
        yield op_desc if block_given?
        op_desc.save
      end

      def placeholder(name='placeholder', dtype=:int32)
        self.create_operation('Placeholder', name) do |op_desc|
          op_desc.attr('dtype').dtype = dtype
        end
      end

      def constant(value, name='const')
        tensor = value.is_a?(Tensor) ? value : Tensor.new(value)
        name = self.scoped_name(name)

        self.create_operation('Const', name) do |op_desc|
          op_desc.attr('value').tensor = tensor
          op_desc.attr('dtype').dtype = tensor.dtype
        end
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

      def copy_function(function, gradient=nil)
        Status.check do |status|
          FFI.TF_GraphCopyFunction(self, function, gradient, status)
        end
      end

      def to_function(name, operators, inputs, outputs, output_names)
        inputs_ptr = inputs ? FFI::Output.pointer_array(inputs) : nil
        outputs_ptr = outputs ? FFI::Output.pointer_array(outputs) : nil

        # Check output names size
        if output_names && output_names.length != Array(outputs).length
          raise(ArgumentError, "output_names length must equal outputs length or be nil")
        end

        # Convert to pointers - keep reference to pointers so they are not GC'ed until the end of the method
        output_names_ptr = if output_names
                             output_names_ptrs = output_names.map do |output_name|
                               ::FFI::MemoryPointer.from_string(output_name)
                             end
                             output_names_ptr = ::FFI::MemoryPointer.new(:pointer, output_names_ptrs.length, true)
                             output_names_ptr.write_array_of_pointer(output_names_ptrs)
                             output_names_ptr
                           else
                             nil
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

      def export
        buffer_ptr = FFI.TF_NewBuffer
        Status.check do |status|
          FFI.TF_GraphToGraphDef(self, buffer_ptr, status)
        end

        buffer = FFI::Buffer.new(buffer_ptr)
        string = buffer[:data].read_string(buffer[:length])
        GraphDef.decode(string)
      ensure
        FFI.TF_DeleteBuffer(buffer)
      end

      def import(graph_def, options=nil)
        options ||= GraphDefOptions.new

        data = if graph_def.is_a?(GraphDef)
                 GraphDef.encode(graph_def)
               else
                 graph_def
               end

        ptr = ::FFI::MemoryPointer.new(:char, data.bytesize)
        ptr.put_bytes(0, data)

        buffer = FFI::Buffer.new
        buffer[:data] = ptr
        buffer[:length] = data.bytesize

        Status.check do |status|
            FFI.TF_GraphImportGraphDef(self, buffer, options, status)
        end
      end
    end
  end
end