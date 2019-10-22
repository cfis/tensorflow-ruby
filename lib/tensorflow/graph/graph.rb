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

      # Adds a placeholder to the Graph, a placeholder is an
      # operation that must be fed with data on execution.
      # Notice that this does not have the shape parameter.
      # Official documentation of {tf.placeholder}[https://www.tensorflow.org/api_docs/python/io_ops/placeholders#placeholder].
      def placeholder(name, type_enum)
        opspec = Tensorflow::OpSpec.new(name, 'Placeholder', 'dtype' => {type_enum => 'DataType'})
        operation = AddOperation(opspec)
        operation.output(0)
      end

      # Creates a constant Tensor that is added to the graph with a specified name.
      # Official documentation of {tf.constant}[https://www.tensorflow.org/versions/r0.9/api_docs/python/constant_op.html#constant].
      def constant(value, name: nil, dtype: nil)
        # Value is the tensor but for now we can ignore that shit
        # Raise error if name and data type are incorrect in any way
        # we have both datatype and tensor for this.
        tensor = Tensorflow::Tensor.new(value, dtype)
        name ||= default_name('Constant')
        opspec = Tensorflow::OpSpec.new(name, 'Const', 'dtype' => {tensor.type_num => 'DataType' }, 'value' => {tensor => 'tensor'})
        operation = AddOperation(opspec)
        operation.output(0)
      end

      # Add a method for variables so that they are not alone
      # everything uptil set attributes is okay but then we need reflect equivalent for ruby
      def AddOperation(opspec)
        opspec.name = opspec.type if opspec.name.nil?
        opspec.name = opspec.type if opspec.name == ''
        cname = CString(opspec.name)
        ctype = CString(opspec.type)
        cdesc = Tensorflow::TF_NewOperation(c, ctype, cname)

        unless opspec.input.empty?
          opspec.input.each do |name|
            Tensorflow::TF_AddInput(cdesc, name.c)
          end
        end

        unless opspec.inputlist.empty?
          c_array = Tensorflow::TF_Output_vector.new
          length = opspec.inputlist.length
          opspec.inputlist.each_with_index { |value, i| c_array[i] = value.c }
          c_array = Tensorflow::TF_Output_array_from_vector(c_array)
          cdesc = Tensorflow.input_list_helper(cdesc, c_array, length)
        end

        status = Tensorflow::Status.new
        opspec.attr.each do |name, value|
          cdesc, status = set_attributes(cdesc, status, name, value)
          # Memory leak here as the TF_OperationDescription
          # object will not be cleaned up. At the time of this
          # writing, this was next to impossible since it
          # required value to be a string tensor with
          # incorrectly encoded strings. Given this rarity, live
          # with the memory leak.  If it becomes a real problem,
          # consider adding a TF_DeleteOperationDescription
          # function to the C API.
        end
        Tensorflow::Operation.new(Tensorflow::TF_FinishOperation(cdesc, status.c), self)
      end

      private
      # Setting attributes is a complicated process for ruby and could have been much
      # more convinient and automated if ruby had run-time reflection like golang.
      # Basically its not possible to differentiate between int32 and int64
      # or float32 and double(float64). This is why attribute specification has been done in the following way.
      # Make a hash of Attributes
      # With the key as the name of the attribute and the value as a hash of one object.
      # The first index of the array is the value itself and the second is the type of the attributes.
      # You can find the types of the attributes on this link https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ops.pbtxt
      # This API is Currently being improved feel free to raise an issue or ask clarification about any query regarding this.
      #
      def set_attributes(cdesc, status, name, value)
        cAttrName = CString(name)
        if value.is_a?(Hash)
          value, type = value.first
        end
        # Some defaults types for attributes of given name
        type = 'DataType'      if name == 'dtype'
        type = 'Tensor'        if name == 'value'
        type = 'int64'         if name == 'channels'
        type = 'DataType'      if name == 'DstT'
        type = 'int32_array'   if name == 'size/Const'

        if value.is_a?(Hash)
          value, type = value.first
        end
        case type
          when 'string'
            Tensorflow::TF_SetAttrString(cdesc, cAttrName, CString(value), value.length)
          when 'string_array'
            size = value.length
            c_string_vector = Tensorflow::String_Vector.new
            list = Tensorflow::Long_long.new
            value.each_with_index do |string, index|
              c_string_vector[index] = string
              list[index] = string.length
            end
            c_array = string_array_from_string_vector(c_string_vector)
            Tensorflow::TF_SetAttrString(cdesc, cAttrName, c_array, list, value.length)
          when 'int32'
            Tensorflow::TF_SetAttrInt(cdesc, cAttrName, value)
          when 'int32_array'
            size = value.length
            list = Tensorflow::Int.new
            value.each_with_index do |number, index|
              c_string_vector[index] = number
            end
            Tensorflow::TF_SetAttrIntList(cdesc, cAttrName, list, size)
          when 'int64'
            Tensorflow::TF_SetAttrInt(cdesc, cAttrName, value)
          when 'int64_array'
            size = value.length
            list = Tensorflow::Long_long.new
            value.each_with_index do |number, index|
              c_string_vector[index] = number
            end
            Tensorflow::TF_SetAttrIntList(cdesc, cAttrName, list, size)
          when 'float32'
            Tensorflow::TF_SetAttrFloat(cdesc, cAttrName, value)
          when 'float32_array'
            size = value.length
            list = Tensorflow::Float.new
            value.each_with_index do |number, index|
              c_string_vector[index] = number
            end
            Tensorflow::TF_SetAttrFloatList(cdesc, cAttrName, list, size)
          when 'DataType'
            Tensorflow::TF_SetAttrType(cdesc, cAttrName, value)
          when 'Tensor'
            Tensorflow::TF_SetAttrTensor(cdesc, cAttrName, value.tensor, status.c)
          # TODO: Insert Tensor_list, DataType_list, Bool
          else
            raise 'Attribute type not supported or attribute type not specififed properly. Please look into the documentation for set_attributes in the graph class for more information.'
        end
        # Shapes can be done, but will require that it be
        # distinguishable from []int64. Which is fine, it
        # probably makes sense to define a Shape type anyway,
        # since that should handle partially known shapes as
        # well and hide the special meaning of -1?
        [cdesc, status]
      end

      # Returns a default name for a new variable or constant.
      # The name increments for each one created: Variable:0, Variable:1, and so on.
      def default_name(type)
        name = "#{type}_#{@number_of_defaults_created[type]}"
        @number_of_defaults_created[type] += 1
        name
      end
    end
  end
end