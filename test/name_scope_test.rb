require_relative 'test_helper'

module Tensorflow
  class NameScopeTest < Minitest::Test
    def setup
      Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
      Graph::Graph.reset_default
    end

    def test_unique_name
      scope = NameScope.new

      assert_equal('foo_2', scope.unique_name('foo_2'))
      assert_equal('foo', scope.unique_name('foo'))
      assert_equal('foo_1', scope.unique_name('foo'))
      assert_equal('foo_3', scope.unique_name('foo'))
    end

    def test_unique_name_case_insensitive
      scope = NameScope.new

      assert_equal('foo', scope.unique_name('foo'))
      assert_equal('Foo_1', scope.unique_name('Foo'))
    end

    def test_unique_constant_name
      constant_1 = Tensorflow.constant(5.0, name: 'c')
      assert_equal('c', constant_1.name, 'c')

      constant_2 = Tensorflow.constant(5.0, name: 'c')
      assert_equal('c_1', constant_2.name)
    end

    def test_scoped_name_case_insensitive
      scope = NameScope.new

      scope.name_scope('bar') do
        assert_equal('bar/foo', scope.scoped_name('foo'))
      end

      scope.name_scope('Bar') do
        assert_equal('Bar_1/foo', scope.scoped_name('foo'))
      end
    end

    def test_no_scope
      constant = Tensorflow.constant(5.0)
      assert_equal('Const', constant.name)
    end

    def test_scope_block
      ExecutionContext.context.current.name_scope('nested') do |scope|
        assert_equal('nested', scope)
      end
    end

    def test_scope
      ExecutionContext.context.current.name_scope('nested') do
        constant = Tensorflow.constant(5.0, name: 'c')
        assert_equal('nested/c', constant.name)
      end
    end

    def test_nested_scopes
      ExecutionContext.context.current.name_scope('nested') do
        ExecutionContext.context.current.name_scope('inner') do
          constant = Tensorflow.constant(5.0, name: 'c')
          assert_equal('nested/inner/c', constant.name)
        end

        ExecutionContext.context.current.name_scope('inner') do
          constant = Tensorflow.constant(5.0, name: 'c')
          assert_equal('nested/inner_1/c', constant.name)
        end
      end
    end

    def test_reset_scope
      ExecutionContext.context.current.name_scope('nested') do
        constant = Tensorflow.constant(5.0, name: 'c')
        assert_equal('nested/c', constant.name)
        ExecutionContext.context.current.name_scope(nil) do |scope|
          refute(scope)
          constant = Tensorflow.constant(5.0, name: 'c')
          assert_equal('c', constant.name)
        end
        constant = Tensorflow.constant(5.0, name: 'c')
        assert_equal('nested/c_1', constant.name)
      end
    end
  end
end