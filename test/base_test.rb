require "bundler/setup"
require "minitest/autorun"
require "tensorflow"
require "tensorflow/extensions/array"

module Tensorflow
  class BaseTest < Minitest::Test
    def eager_and_graph(&block)
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
          next_element = iterator.get_next

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

    def result_context(context, operation)
      case operation
        when Data::Dataset
          operation.data
        when Eager::TensorHandle
          operation.value
      end
    end
  end
end
