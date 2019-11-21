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
        loss = NN.sparse_softmax_cross_entropy_with_logits(logits, labels)
      end

      def optimizer(loss)
        optimizer = Train::GradientDescentOptimizer.new(self.learning_rate)
        global_step = Variable.new(0, name: 'global_step', trainable: false)
        optimizer.minimize(loss, global_step: global_step)
      end

      def evaluation(logits, labels)
        correct = Math.in_top_k(logits, labels, k: 1)
        Tf.reduce_sum(Tf.cast(correct, :int32))
      end

      def model(images, hidden_1_units, hidden_2_units)
        # Hidden layer 1
        hidden_1 = Tf.name_scope('hidden_1') do
          stddev = 1.0/::Math.sqrt(IMAGE_PIXELS)
          normal = Random.normal([IMAGE_PIXELS, hidden_1_units], stddev: stddev)
          weights = Variable.new(normal, name: 'weights')

          zeros = Tf.zeros([hidden_1_units])
          biases = Variable.new(zeros, name: 'biases')

          matmul = Linalg.matmul(images, weights)
          NN.relu(matmul + biases)
        end

        # Hidden layer 2
        hidden_2 = Tf.name_scope('hidden_2') do
          stddev = 1.0/::Math.sqrt(hidden_1_units)
          normal = Random.normal([hidden_1_units, hidden_2_units], stddev: stddev)
          weights = Variable.new(normal, name: 'weights')

          zeros = Tf.zeros([hidden_2_units])
          biases = Variable.new(zeros, name: 'biases')

          matmul = Linalg.matmul(hidden_1, weights)
          NN.relu(matmul + biases)
        end

        # Linear
        Tf.name_scope('softmax_linear') do
          stddev = ::Math.sqrt(hidden_2_units)
          normal = Random.normal([hidden_2_units, NUM_CLASSES], stddev: stddev)
          weights = Variable.new(normal, name: 'weights')

          zeros = Tf.zeros([NUM_CLASSES])
          biases = Variable.new(zeros, name: 'biases')

          matmul = Linalg.matmul(hidden_2, weights)
          NN.relu(matmul + biases)
        end
      end

      def train
        Graph::Graph.default.as_default do |graph|
          ds = self.dataset.train.batch(1000)
          iterator = ds.make_initializable_iterator
          next_element = iterator.get_next
          #training_init = iterator.make_initializer(self.dataset.train)
          #training_init = iterator.make_initializer(self.dataset.train)

          logits = self.model(next_element[0], 128, 32)
          loss = self.loss(logits, next_element[1])
          optimizer = self.optimizer(loss)
          eval_correct = self.evaluation(logits, next_element[1])

          Graph::Session.run(graph) do |session|
            session.run(Tf.global_variables_initializer)
            session.run(iterator.initializer)

            self.max_steps.times do |step|
              result = session.run([loss, optimizer])
              # if i % 50 == 0:
              #     print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))
              # now setup the validation run
              #
              # Write the summaries and print an overview fairly often.
              if step % 100 == 0
                STDOUT << "Step #{step}: loss = #{result}" << "\n"
              end
            end
          end
        end
      end
    end
  end
end

mnist = Tf::Samples::Mnist.new
mnist.train