require_relative "../test_helper"
require 'stringio'

module Tensorflow
  module Printers
    class GraphPrinterTest < Minitest::Test
      def test_print
        graph = Tensorflow::Graph::Graph.new
        x = graph.constant(3.0, "x")
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
                    []
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
                    []
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
        EOS

        assert_equal(expected, io.string)
      end
    end
  end
end
