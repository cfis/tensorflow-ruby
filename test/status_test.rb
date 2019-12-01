require_relative "base_test"

module Tensorflow
  class StatusTest < BaseTest
    def test_default_status
      status = Status.new
      assert_equal(:tf_ok, status.code)
      assert_empty(status.message)
    end

    def test_set_status
      status = Status.new
      status.set(:tf_cancelled, 'Status is cancelled')
      assert_equal(:tf_cancelled, status.code)
      assert_equal('Status is cancelled', status.message)
    end

    def test_valid_status
      graph = Graph::Graph.new
      buffer = FFI::Buffer.new
      status = Status.new
      FFI.TF_GraphGetOpDef(graph, "Variable", buffer, status)

      assert_equal(:tf_ok, status.code)
      assert_empty(status.message)
    end

    def test_invalid_status
      graph = Graph::Graph.new
      ptr = ::FFI::MemoryPointer.new(:pointer)
      status = Status.new
      buffer = FFI::Buffer.new
      FFI.TF_GraphGetOpDef(graph, "NotAOp", buffer, status)

      assert_equal(:tf_not_found, status.code)
      assert_match(/^Op type not registered 'NotAOp' in binary running on.*Make sure the Op and Kernel are registered in the binary running in this process/, status.message)
    end

    def test_check
      graph = Graph::Graph.new
      ptr = ::FFI::MemoryPointer.new(:pointer)
      status = Status.new
      buffer = FFI::Buffer.new
      FFI.TF_GraphGetOpDef(graph, "NotAOp", buffer, status)

      error = assert_raises(Error::NotFoundError) do
        status.check
      end
      assert_match(/^Op type not registered 'NotAOp' in binary running on.*Make sure the Op and Kernel are registered in the binary running in this process/, error.message)
    end

    def test_class_check
      graph = Graph::Graph.new
      ptr = ::FFI::MemoryPointer.new(:pointer)
      buffer = FFI::Buffer.new

      error = assert_raises(Error::NotFoundError) do
        Status.check do |status|
          FFI.TF_GraphGetOpDef(graph, "NotAOp", buffer, status)
        end
      end
      assert_match(/^Op type not registered 'NotAOp' in binary running on.*Make sure the Op and Kernel are registered in the binary running in this process/, error.message)
    end
  end
end