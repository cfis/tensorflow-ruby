module Tensorflow
  module Graph
    class Graph
      attr_reader :control_inputs

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
        @collections = Hash.new
        @name_scope = NameScope.new
        @pointer = FFI.TF_NewGraph()
        @control_inputs = Array.new
        ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
      end

      def to_ptr
        @pointer
      end

      def collections
        @collections.keys
      end

      def add_to_collection(name, value)
        values = @collections[name] ||= Array.new
        values << value
      end

      def add_to_collections(names, value)
        names.each do |name|
          self.add_to_collection(name, value)
        end
      end

      def get_collection_ref(name, scope=nil)
        @collections[name]
      end

      def clear_collection(name)
        @collections[name] = Array.new
      end

      def as_default
        raise(Error::InvalidArgumentError, "Must provide block") unless block_given?
        ExecutionContext.push(self)
        begin
          yield self
        ensure
          ExecutionContext.pop
        end
      end

      def control_dependencies(control_inputs)
        @control_inputs = Array(control_inputs)
        begin
          yield self
        ensure
          @control_inputs = []
        end
      end

      def op_def(op_type)
        buffer_ptr = FFI.TF_NewBuffer
        Status.check do |status|
          FFI.TF_GraphGetOpDef(self, op_type, buffer_ptr, status)
        end
        buffer = FFI::Buffer.new(buffer_ptr)
        string = buffer[:data].read_string(buffer[:length])
        OpDef.decode(string)
      ensure
        FFI.TF_DeleteBuffer(buffer)
      end

      def forward(operation)
        def forward_internal(set, operation)
          operation.consumers.each do |consumer|
            set << consumer.operation
            forward_internal(set, consumer.operation)
          end
          set
        end
        result = Set.new([operation])
        forward_internal(result, operation)
      end

      def backward(operation)
        def backward_internal(set, operation)
          operation.inputs.each do |input|
            set << input.operation
            backward_internal(set, input.operation)
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

      def output_shapes(operation)
        operation.outputs.map do |output|
          num_dims = Status.check do |status|
            FFI.TF_GraphGetTensorNumDims(self, output, status)
          end

          if num_dims == -1
            []
          else
            dims_ptr = ::FFI::MemoryPointer.new(:int64, num_dims)
            Status.check do |status|
              FFI.TF_GraphGetTensorShape(self, output, dims_ptr, num_dims, status)
            end
            dims_ptr.read_array_of_int64(num_dims)
          end
        end
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

      def add_function(function, gradient=nil)
        Status.check do |status|
          FFI.TF_GraphCopyFunction(self, function, gradient, status)
        end
      end

      def to_function(name, operators, input_operations, output_operations, output_names=nil)
        inputs = input_operations ? input_operations.map(&:outputs).flatten : []
        inputs_ptr = FFI::Output.array_to_ptr(inputs.map(&:output))

        outputs = output_operations ? output_operations.map(&:outputs).flatten : []
        outputs_ptr = FFI::Output.array_to_ptr(outputs.map(&:output))

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
        output_types = output_operations.map(&:output_types).flatten(1)
        output_shapes = output_operations.map(&:output_shapes).flatten(1)
        Function.new(func, output_types, output_shapes)
      end

      def as_graph_def
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