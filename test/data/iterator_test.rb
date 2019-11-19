require_relative "../base_test"

module Tensorflow
  module Data
    class IteratorTest < BaseTest
      def test_reinitializable
        Graph::Graph.new.as_default do |graph|
          const_1 = Tensorflow.constant([1, 2, 3])
          const_2 = Tensorflow.constant([4, 5, 6, 7])
          dataset_1 = Dataset.from_tensors(const_1)
          dataset_2 = Dataset.from_tensors(const_2)

          iterator = InitializableIterator.new(dataset_1.output_types, [[-1]])
          get_next = iterator.get_next
          dataset_1_init_op = iterator.make_initializer(dataset_1)
          dataset_2_init_op = iterator.make_initializer(dataset_2)

          assert_equal(dataset_1.output_types, iterator.output_types)
          assert_equal(dataset_2.output_types, iterator.output_types)

          Graph::Session.run(graph) do |session|
            session.run(dataset_1_init_op)
            assert_equal([1, 2, 3], session.run(get_next))

            session.run(dataset_2_init_op)
            assert_equal([4, 5, 6, 7], session.run(get_next))

            session.run(dataset_1_init_op)
            assert_equal([1, 2, 3], session.run(get_next))
          end
        end
      end
    end
  end
end
