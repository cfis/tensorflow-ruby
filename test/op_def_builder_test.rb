require_relative "test_helper"

module Tensorflow
  class OpDefBuilderTest < Minitest::Test
    def test_basic
      builder = OpDefBuilder.new('SomeOp')
      builder.attr("attr1: string")
      builder.input("input1: uint8")
      builder.input("input2: uint16")
      builder.output("output1: uint32")
      builder.register

      op_def = Tensorflow.op_def('SomeOp')
      assert(op_def)

      assert_equal(2, op_def.input_arg.length)
      input_arg = op_def.input_arg[0]
      assert_equal("input1", input_arg.name)
      assert_equal(:DT_UINT8, input_arg.type)

      input_arg = op_def.input_arg[1]
      assert_equal("input2", input_arg.name)
      assert_equal(:DT_UINT16, input_arg.type)

      assert_equal(1, op_def.attr.length)
      attr = op_def.attr[0]
      assert_equal("attr1", attr.name)
      assert_equal("string", attr.type)

      assert_equal(1, op_def.output_arg.length)
      output_arg = op_def.output_arg[0]
      assert_equal("output1", output_arg.name)
      assert_equal(:DT_UINT32, output_arg.type)
    end

    def test_shape_inference
      OpDefBuilder.new("TestShapeInference")
            .attr("attr1: tensor")
            .shape_inference(OpDefBuilder.unknown_shape_inference_func)
            .register

      op_def = Tensorflow.op_def('TestShapeInference')
      assert(op_def)

      assert_equal(0, op_def.input_arg.length)
      assert_equal(1, op_def.attr.length)
      attr = op_def.attr[0]
      assert_equal("attr1", attr.name)
      assert_equal("tensor", attr.type)

      assert_equal(0, op_def.output_arg.length)
    end
  end
end