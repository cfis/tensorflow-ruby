require_relative "../test_helper"
require 'stringio'

module Tensorflow
  module Printers
    class GraphPrinterTest < Minitest::Test
      def setup
        Tensorflow.execution_mode = Tensorflow::EAGER_MODE
      end

      def test_print
        Tensorflow::Graph::Graph.new.as_default do |graph|
          x = Tensorflow.constant(3.0, name: "x")
          y = Math.pow(x, 2.0)

          io = StringIO.new
          printer = Printers::Graph.new(graph)
          printer.print(io)

          expected = <<~EOS
            node {
              name: "x"
              op: "Const"
              attr {
                key: "dtype"
                value {
                  type: DT_FLOAT
                }
              }
              attr {
                key: "value"
                value {
                  tensor {
                    dtype: DT_FLOAT
                    tensor_shape {
                    }
                    float_val: 3.0
                  }
                }
              }
            }
            node {
              name: "Pow/y"
              op: "Const"
              attr {
                key: "dtype"
                value {
                  type: DT_FLOAT
                }
              }
              attr {
                key: "value"
                value {
                  tensor {
                    dtype: DT_FLOAT
                    tensor_shape {
                    }
                    float_val: 2.0
                  }
                }
              }
            }
            node {
              name: "Pow"
              op: "Pow"
              input: "x"
              input: "Pow/y"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            versions {
              producer: 119
            }
          EOS

          assert_equal(expected, io.string)
        end
      end
    end
  end
end
