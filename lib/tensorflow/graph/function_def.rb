module Tensorflow
  module Graph
    class FunctionDef
      attr_reader :ruby_method, :signatures

      Signature = Struct.new(:dtype, :shape)

      def initialize(ruby_method, input_signatures = [])
        @ruby_method = ruby_method
        self.process_signatures(ruby_method, input_signatures)
        self.wrap_ruby_method
      end

      def process_signatures(ruby_method, input_signatures)
        if input_signatures.length != ruby_method.parameters.length
          raise(TensorflowError, "Must specify input signature for each method parameter")
        end

        @signatures = input_signatures.map do |dtype, shape|
          Signature.new(dtype, shape)
        end
      end

      def aliased_name
        "#{self.ruby_method.original_name}_original"
      end

      def wrap_ruby_method
        new_name = self.aliased_name
        original_name = self.ruby_method.original_name
        self.ruby_method.owner.instance_eval do
          alias_method(new_name, original_name)
        end

        this = self
        original_name = ruby_method.original_name
        self.ruby_method.owner.instance_eval do
          define_method(original_name) do |*args|
            function = this.build_function(self)
            ExecutionContext.current.add_function(function)
            function
          end
        end
      end 

      def build_function(object)
        Graph::new.as_default do |graph|
          placeholders = self.ruby_method.parameters.map.with_index do |param, index|
            signature = self.signatures[index]
            Tensorflow.placeholder(signature.dtype, name: param.last, shape: signature.shape)
          end

          # Call the original ruby_method to build the graph
          bound_method = self.ruby_method.bind(object)
          result = bound_method.call(*placeholders)

          graph.to_function(self.ruby_method.original_name.to_s, nil, placeholders, Array(result))
        end
      end
    end
  end
end
