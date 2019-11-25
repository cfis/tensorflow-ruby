$LOAD_PATH.unshift(File.expand_path('../../lib', __dir__))

require "tensorflow"
require_relative "../../lib/datasets/images/mnist"

#Tf.disable_eager_execution

module Tensorflow
  module Samples
    class Mnist
      NUM_CLASSES = 10
      IMAGE_SIZE = 28
      IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

      attr_reader :batch_size, :dataset, :learning_rate, :max_steps

      def initialize
        @batch_size = 100
        @learning_rate = 0.01
        @max_steps = 2_000
        @dataset = Datasets::Images::Mnist.new
      end

      def loss(logits, labels)
        labels = Tf.cast(labels, :int64)
        xentropy = NN.sparse_softmax_cross_entropy_with_logits(logits, labels)
        Math.reduce_mean(xentropy)
      end

      def optimizer(loss)
        optimizer = Train::GradientDescentOptimizer.new(self.learning_rate)
        global_step = Variable.new(0, name: 'global_step', trainable: false)
        optimizer.minimize(loss, global_step: global_step)
      end

      def evaluation(logits, labels)
        correct = Math.in_top_k(logits, labels, 1)
        Tf.reduce_sum(Tf.cast(correct, :int32))
      end

      def model(images, hidden_1_units, hidden_2_units)
        # Hidden layer 1
        hidden_1 = Tf.name_scope('hidden_1') do
          stddev = 1.0/::Math.sqrt(IMAGE_PIXELS)
          normal = Random.truncated_normal([IMAGE_PIXELS, hidden_1_units], stddev: stddev, seed: 10.0)
          weights = Variable.new(normal, name: 'weights')

          zeros = Tf.zeros([hidden_1_units])
          biases = Variable.new(zeros, name: 'biases')

          matmul = Linalg.matmul(images, weights)
          NN.relu(matmul + biases)
        end

        # Hidden layer 2
        hidden_2 = Tf.name_scope('hidden_2') do
          stddev = 1.0/::Math.sqrt(hidden_1_units)
          normal = Random.truncated_normal([hidden_1_units, hidden_2_units], stddev: stddev, seed: 10.0)
          weights = Variable.new(normal, name: 'weights')

          zeros = Tf.zeros([hidden_2_units])
          biases = Variable.new(zeros, name: 'biases')

          matmul = Linalg.matmul(hidden_1, weights)
          NN.relu(matmul + biases)
        end

        # Linear
        Tf.name_scope('softmax_linear') do
          stddev = 1.0/::Math.sqrt(hidden_2_units)
          normal = Random.truncated_normal([hidden_2_units, NUM_CLASSES], stddev: stddev, seed: 10.0)
          weights = Variable.new(normal, name: 'weights')

          zeros = Tf.zeros([NUM_CLASSES])
          biases = Variable.new(zeros, name: 'biases')

          matmul = Linalg.matmul(hidden_2, weights)
          NN.relu(matmul + biases)
        end
      end

      def train(session, iterator, optimizer, loss)
        ds = self.dataset.train.batch(1)
        iterator_initializer = iterator.make_initializer(ds)
        session.run(iterator_initializer)

        step = 0
        start = Time.now

        begin
          loop do
            _, loss_value = session.run([optimizer, loss])
            if step % 1000 == 0
              puts "Step: #{step}, Duration: #{Time.now - start}, loss: #{loss_value}"
            end
            step += 1
          end
        rescue Error::OutOfRangeError => e
          puts e
        end
      end

      def test(session, iterator, accuracy)
        ds = self.dataset.test.batch(1)
        iterator_initializer = iterator.make_initializer(ds)
        session.run(iterator_initializer)

        step = 0
        start = Time.now
        true_count = 0

        begin
          loop do
            true_count += session.run(accuracy)
            if step % 100 == 0
              puts "Step: #{step}, Duration: #{Time.now - start}, true_count: #{true_count}, accuracy: #{true_count/step.to_f}"
            end
            step += 1
          end
        rescue Error::OutOfRangeError => e
          puts e
        end

        STDOUT << "True: " <<  true_count << "\n"
        STDOUT << "Accuracy: " <<  true_count/step.to_f << "\n"
      end

      def run
        Graph::Graph.default.as_default do |graph|
          train_ds = self.dataset.train.batch(1)
          iterator = Data::Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
          next_element = iterator.get_next

          logits = self.model(next_element[0], 128, 32)
          loss = self.loss(logits, next_element[1])
          optimizer = self.optimizer(loss)
          accuracy = self.evaluation(logits, next_element[1])

          Graph::Session.run(graph) do |session|
            session.run(Tf.global_variables_initializer)
            self.train(session, iterator, optimizer, loss)
            self.test(session, iterator, accuracy)
          end
        end
      end
    end
  end
end

mnist = Tf::Samples::Mnist.new
mnist.run