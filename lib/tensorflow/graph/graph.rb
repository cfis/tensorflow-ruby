module Tensorflow
  module Graph
    class Graph
      extend Forwardable
      def_delegators :@name_scope, :name_scope, :scoped_name, :unique_name

      def self.default
        @default ||= Graph.new
      end

      def self.reset_default
        @default = Graph.new
      end

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

      def as_default
        raise(TensorflowError, "Must provide block") unless block_given?
        ExecutionContext.context.push(self)
        begin
          yield self
        ensure
          ExecutionContext.context.pop
        end
      end

      def op_def(name)
        buffer = FFI::Buffer.new
        Status.check do |status|
          FFI.TF_GraphGetOpDef(self, name, buffer, status)
        end
        string = buffer[:data].read_string(buffer[:length])
        ops = OpDef.decode(string)
      end

      def forward(operation)
        def forward_internal(set, operation)
          operation.consumers.each do |consumer|
            set << consumer
            forward_internal(set, consumer)
          end
          set
        end
        result = Set.new([operation])
        forward_internal(result, operation)
      end

      def backward(operation)
        def backward_internal(set, operation)
          operation.inputs.each do |input|
            input_operation = input.operation(self)
            set << input_operation
            backward_internal(set, input_operation)
          end
          set
        end
        result = Set.new([operation])
        backward_internal(result, operation)
      end

      def operations
        return enum_for(:operations) unless block_given?

        # Get a pointer to a size_t set to 0
        position_ptr = ::FFI::MemoryPointer.new(:size_t, 1, true)
        while (ptr = FFI.TF_GraphNextOperation(self, position_ptr))
          break if ptr.null?
          yield Operation.new(self, ptr)
        end
      end

      def operation(name)
        ptr = FFI.TF_GraphOperationByName(self, name)
        ptr.null? ? nil : Operation.new(self, ptr)
      end

      def create_operation(op_type, inputs=[], attrs={})
        op_desc = OperationDescription.new(self, op_type, inputs, attrs)
        op_desc.save
      end

      def execute(operations, feed_dict={})
        session = Session.new(self, SessionOptions.new)
        result = session.run(operations, feed_dict)
        session.close
        result
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

      def to_function(name, operators, input_operations, output_operations, output_names)
        inputs = input_operations ? input_operations.map(&:outputs).flatten : []
        inputs_ptr = FFI::Output.array_to_ptr(inputs)

        outputs = output_operations ? output_operations.map(&:outputs).flatten : []
        outputs_ptr = FFI::Output.array_to_ptr(outputs)

        # Check output names size
        if output_names && output_names.length != outputs.length
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