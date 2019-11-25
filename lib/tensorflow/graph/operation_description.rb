module Tensorflow
  module Graph
    class OperationDescription
      attr_reader :graph, :name, :op_def

      def initialize(graph, op_type, inputs, attrs)
        @graph = graph
        @op_def = case op_type
                    when Function
                      op_type.function_def.signature
                    else
                      self.graph.op_def(op_type)
                  end
        raise(Error::InvalidArgumentError, "Invalid op type: #{op_type}") unless @op_def

        raw_name = attrs.delete(:name)&.to_s || self.op_def.name
        @name = self.graph.scoped_name(raw_name)
        @pointer = FFI.TF_NewOperation(graph, self.op_def.name, @name)

        inputs = Array(inputs)
        setup_inputs(inputs, attrs)
        setup_control_inputs(graph.control_inputs)
        setup_attrs(**attrs)
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

      def setup_control_inputs(control_inputs)
        control_inputs.each do |control_input|
          setup_control_input(control_input)
        end
      end

      def setup_control_input(control_input)
        control_input = case control_input
                          when Operation
                            control_input
                          when Variable
                            control_input.handle
                          else
                            raise(Error::InvalidArgumentError, "Invalid control input")
                          end

        FFI.TF_AddControlInput(self, control_input)
      end

      def capture_inputs(operation, attrs)
        # First capture the inputs
        inputs = operation.inputs.map do |input|
          self.capture(input.operation)
        end

        # We now have to group the inputs together. For example, a TensorSlice dataset has 1 input argument
        # which a list. But the number of inputs returned by the operation is actually the number of items in
        # the list, so its usually more than one. We need to group them into one array to be able to call
        # the operation to create a captured copy.
        i = 0
        operation.op_def.input_arg.reduce(Array.new) do |result, input_arg|
          if !input_arg.number_attr.empty?
            input_len = attrs[input_arg.number_attr.to_sym]
            is_sequence = true
          elsif !input_arg.type_list_attr.empty?
            input_len = attrs[input_arg.type_list_attr.to_sym].length
            is_sequence = true
          else
            input_len = 1
            is_sequence = false
          end

          if is_sequence
            result << inputs[i..i+input_len]
          else
            result << inputs[i]
          end
          i += input_len
          result
        end
      end

      def capture(operation)
        if self.op_def.is_stateful
          raise(Error::InvalidArgumentError, "Cannot capture a stateful node (name: #{operation.name}, type: #{operation.op_type})")
        elsif operation.op_type == "Placeholder"
          raise(Error::InvalidArgumentError, "Cannot capture a placeholder by value (name: #{operation.name}, type: #{operation.op_type})")
        end

        attrs = operation.attributes.reduce(Hash.new) do |hash, attr|
          hash[attr.name.to_sym] = attr.value
          hash
        end
        attrs[:name] = operation.name

        captured_inputs = self.capture_inputs(operation, attrs)
        self.graph.create_operation(operation.op_type, captured_inputs, **attrs)
      end

      def check_input(arg_def, input, dtype)
        case input
          when Operation
            self.graph.equal?(input.graph) ? input : capture(input)
          when OperationOutput
            input
          when FFI::Output
            raise(Error::UnknownError, "shouldn't get here")
          when Variable
            arg_def.type == :DT_RESOURCE ? input.handle : input.value_handle
          else
            input_name = "#{self.name}/#{arg_def.name}"
            Tensorflow.constant(input, name: input_name, dtype: dtype)
        end
      end

      def setup_inputs(inputs, attrs)
        inputs.each_with_index do |input, index|
          self.setup_input(index, input, attrs)
        end
      end

      def setup_input(index, value, attrs)
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
        elsif !arg_def.number_attr.empty?
          # This input is a homogeneous list
          self.add_input_list(checked_value)
        else
          # This input is a single item
          self.add_input(checked_value)
        end
      end

      def add_input(operation)
        # Check to see if the operation has multiple outputs, and if it does, we need to pack them together
        # to fit into one input
        if operation.is_a?(OperationOutput)
          FFI.TF_AddInput(self, operation)
        elsif operation.num_outputs > 1
          packed = Tensorflow.pack(operation, n: operation.num_outputs)
          FFI.TF_AddInput(self, packed.outputs.first)
        else
          FFI.TF_AddInput(self, operation.outputs.first)
        end
      end

      def add_input_list(operations)
        # Operation can represent multiple operations *or* one operation with multiple outputs (like SPLIT)
        outputs = Array(operations).map(&:outputs).flatten
        outputs_ptr = FFI::Output.array_to_ptr(outputs.map(&:output))
        FFI.TF_AddInputList(self, outputs_ptr, outputs.length)
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
          raise(Error::UnknownError, "Unknown attribute: #{name}")
        end

        case attr_def.type
          when 'bool'
            FFI.TF_SetAttrBool(self, attr_def.name, value ? 1 : 0)
          when 'int'
            FFI.TF_SetAttrInt(self, attr_def.name, value)
          when 'float'
            FFI.TF_SetAttrFloat(self, attr_def.name, value)
          when 'func'
            function_name = value.is_a?(Function) ? value.name : value
            FFI.TF_SetAttrFuncName(self, attr_def.name, function_name, function_name.length)
          when 'shape'
            pointer = ::FFI::MemoryPointer.new(:int64, value.length)
            pointer.write_array_of_int64(value)
            FFI.TF_SetAttrShape(self, attr_def.name, pointer, value.length)
          when 'list(shape)'
            dims_pointer = ::FFI::MemoryPointer.new(:pointer, value.length)
            num_dims_pointer = ::FFI::MemoryPointer.new(:int32, value.length)
            value.each_with_index do |shape, i|
              dim_pointer = ::FFI::MemoryPointer.new(:int64, shape.length)
              dim_pointer.write_array_of_int64(shape)
              dims_pointer.put_pointer(i * ::FFI.type_size(:pointer), dim_pointer)
              num_dims_pointer.put_int32(i * ::FFI.type_size(:int32), shape.length)
            end
            FFI.TF_SetAttrShapeList(self, attr_def.name, dims_pointer, num_dims_pointer, value.length)
          when 'string'
            FFI.TF_SetAttrString(self, attr_def.name, value, value.length)
          when 'list(string)'
            a = 1
            #FFI.TF_SetAttrString(self, attr_def.name, value, value.length)
          when 'tensor'
            Status.check do |status|
              FFI.TF_SetAttrTensor(self, attr_def.name, value, status)
            end
          when 'type'
            FFI.TF_SetAttrType(self, attr_def.name, value)
          when 'list(type)'
            value_ptr = ::FFI::MemoryPointer.new(FFI::DataType.native_type.size, value.count)
            value.each_with_index do |a_value, i|
              value_ptr.put_int32(i * FFI::DataType.native_type.size, FFI::DataType[a_value])
            end
            FFI.TF_SetAttrTypeList(self, attr_def.name, value_ptr, value.count)
          else
            raise(Error::UnimplementedError, "Unsupported attribute. #{self.op_def.name} - #{attr_def.name}")
        end
      end
    end
  end
end