require "bundler/setup"
require "minitest/autorun"
require "tensorflow"
require "tensorflow/extensions/array"

module Tensorflow
  class BaseTest < Minitest::Test
    attr_reader :session

    def eager_and_graph(&block)
      self.eager_mode(&block)
      self.graph_mode(&block)
    end

    def eager_mode(&block)
      [Eager::Context.new].each do |context|
        context.as_default do
          yield context
        end
      end
    end

    def graph_mode(&block)
      [Graph::Graph.new].each do |context|
        @session = Graph::Session.new(context, Graph::SessionOptions.new)
        context.as_default do
          yield context
        end
      ensure
        @session&.close
        @session = nil
      end
    end

    def evaluate(operation)
      if self.session
        result_graph(operation)
      else
        result_context(operation)
      end
    end

    def result_graph(operation)
      case operation
        when Array
          self.session.run(operation)
        when Graph::Operation
          self.session.run(operation)
        when Data::Dataset
          iterator = operation.make_initializable_iterator
          next_element = iterator.get_next

          self.session.run(iterator.initializer)
          result = []
          begin
            loop do
              result << self.session.run(next_element)
            end
          rescue Error::OutOfRangeError
            return result
          end
      end
    end

    def result_context(result)
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
