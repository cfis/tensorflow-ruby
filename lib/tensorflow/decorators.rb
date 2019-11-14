module Tensorflow
  module Decorator
    class Function
      attr_reader :input_signatures

      def initialize(input_signatures = [])
        @input_signatures = input_signatures
      end

      def wrap(method)
        Graph::FunctionDef.new(method, self.input_signatures)
      end
    end

    def self.extended(klass)
      @waiting_for_method = false
      this = self
      klass.instance_eval do
        @tf = this
      end

      if klass.is_a?(Object) && klass.to_s == 'main'
        klass.class.extend(self)
      end
    end

    def self.function(input_signature = [])
      @current_function = Function.new(input_signature)
    end

    def self.wrap_method(method)
      # We do this little dance because when the method is wrapped it will trigger method_added. So first we need
      # to clear out @current_function before continuing
      if @current_function
        current_function = @current_function
        @current_function = nil
        current_function&.wrap(method)
      end
    end

    def singleton_method_added(method_name)
      super(method_name)
      method = self.method(method_name)
      @tf.wrap_method(method)
    end

    def method_added(method_name)
      super(method_name)
      method = self.instance_method(method_name)
      @tf.wrap_method(method)
    end
  end
end