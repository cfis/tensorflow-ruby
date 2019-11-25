require "bundler/setup"
require "minitest/autorun"
require "tensorflow"
require "tensorflow/extensions/array"

module Tensorflow
  class BaseTest < Minitest::Test
    def eager_and_graph(&block)
      #[Eager::Context.new].each do |context|
      #[Graph::Graph.new].each do |context|
      [Eager::Context.new, Graph::Graph.new].each do |context|
        context.as_default do
          yield context
        end
      end
    end

    def result(context, operation)
      case context
        when Graph::Graph
          result_graph(context, operation)
        when Eager::Context
          result_context(context, operation)
        end
    end

    def result_graph(context, operation)
      session = Graph::Session.new(context, Graph::SessionOptions.new)

      case operation
        when Graph::Operation
          session.run(operation)
        when Data::Dataset
          iterator = operation.make_one_shot_iterator
          #iterator = operation.make_initializable_iterator
          next_element = iterator.get_next

          #session.run(iterator.initializer)
          result = []
          while true
            begin
              result << session.run(next_element)
            rescue TensorflowError => exception
              return result
            end
          end
      end
    ensure
      session.close
    end

    def result_context(context, result)
      case result
        when Data::Dataset
          result.data
        when Eager::TensorHandle
          result.value
        when Array
          result.map do |sub_result|
            sub_result.value
          end
      end
    end
  end
end
