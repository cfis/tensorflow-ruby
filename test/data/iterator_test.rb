require_relative "../base_test"

module Tensorflow
  module Data
    class IteratorTest < BaseTest
      def test_one_shot
        Graph::Graph.new.as_default do |graph|
          dataset1 = TensorSliceDataset.new(Numo::NArray[1, 2, 3])
          dataset2 = TensorSliceDataset.new(Numo::NArray['a', 'b', 'c'])
          dataset3 = ZipDataset.new(dataset1, dataset2)

          iterator = dataset3.make_one_shot_iterator
          next_element = iterator.get_next

          result = []
          Graph::Session.run(graph) do |session|
            begin
              loop do
                result << session.run(next_element)
              end
            rescue Error::OutOfRangeError
            end
          end
          assert_equal([[1, "a"], [2, "b"], [3, "c"]], result)
        end
      end

      def test_one_shot_split
        Graph::Graph.new.as_default do |graph|
          dataset1 = TensorSliceDataset.new(Numo::NArray[1, 2, 3])
          dataset2 = TensorSliceDataset.new(Numo::NArray['a', 'b', 'c'])
          dataset3 = ZipDataset.new(dataset1, dataset2)

          iterator = dataset3.make_one_shot_iterator
          next_element = iterator.get_next

          result_1 = []
          result_2 = []
          Graph::Session.run(graph) do |session|
            begin
              loop do
                result_1 << session.run(next_element[0])
                result_2 << session.run(next_element[1])
              end
            rescue Error::OutOfRangeError
            end
          end
          assert_equal([1, 3], result_1)
          assert_equal(["b"], result_2)
        end
      end

      def test_initializable
        components = [1,
                      Numo::Int64[1, 2, 3],
                      37.0]

        Graph::Graph.new.as_default do |graph|
          dataset = TensorDataset.new(components)

          iterator = dataset.make_initializable_iterator(shared_name: 'shared_iterator')
          next_element = iterator.get_next

          result = nil
          Graph::Session.run(graph) do |session|
            session.run(iterator.initializer)
            result = session.run(next_element)
          end
          assert_equal([1, [1, 2, 3], 37.0], result)
        end
      end

      def test_reinitializable
        Graph::Graph.new.as_default do |graph|
          const_1 = Tensorflow.constant([1, 2, 3])
          const_2 = Tensorflow.constant([4, 5, 6, 7])
          dataset_1 = Dataset.from_tensors(const_1)
          dataset_2 = Dataset.from_tensors(const_2)

          iterator = ReinitializableIterator.new(dataset_1.output_types, [[-1]])
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
