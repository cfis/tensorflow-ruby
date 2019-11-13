module Tensorflow
  module Decorator
    class Function
      attr_reader :waiting_for_method

      def initialize
        @waiting_for_method = false
      end

      def function
        @waiting_for_method = true
      end

      def wrap_method(method)
        return unless self.waiting_for_method
        @waiting_for_method = false

        new_name = "#{method.original_name}_original"
        method.owner.instance_eval do
          alias_method(new_name, method.original_name)
        end

        method.owner.instance_eval do
          define_method(method.original_name) do |*args|
            current_graph = Tensorflow::ExecutionContext.current
            Tensorflow::Graph::Graph.new.as_default do |function_graph|
              placeholders = method.parameters.map do |flag, name|
                Tensorflow.placeholder(:int32, name: name)
              end
              # Call the original method to build the graph
              result = method(new_name).call(*placeholders)
              # Now convert the graph to a function
              new_function = function_graph.to_function(method.original_name.to_s, nil, placeholders, [result])
              # Add it to the current graph
              current_graph.add_function(new_function)
              new_function
            end
          end
        end
      end
    end

    def self.extended(klass)
      @waiting_for_method = false
      this = self

      klass.instance_eval do
        @tf = Decorator::Function.new
      end

      if klass.is_a?(Object) && klass.to_s == 'main'
        klass.class.extend(self)
      end
    end

    def singleton_method_added(method_name)
      super(method_name)
      method = self.method(method_name)
      @tf&.wrap_method(method)
    end

    def method_added(method_name)
      super(method_name)
      method = self.instance_method(method_name)
      @tf&.wrap_method(method)
    end
  end
end