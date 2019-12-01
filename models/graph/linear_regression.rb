$LOAD_PATH.unshift(File.expand_path('../../lib', __dir__))

require "tensorflow"
require 'pathname'

# Use graph mode execution
Tf.disable_eager_execution

# Create training data
train_x = 10 * Numo::Float32.new([100]).rand
train_y = 4 + 3 * train_x + Numo::Float32.new([100]).rand

# Convert to arrays
train_x = train_x.to_a
train_y = train_y.to_a

# Placeholders
x_value = Tf.placeholder(:float, name: 'x_value')
y_value = Tf.placeholder(:float, name: 'y_value')

# Set model weights
weight = Tf::Variable.new(rand, name: 'weight', trainable: true)
bias = Tf::Variable.new(rand, name: 'bias', trainable: true)

# Construct a linear model
pred = x_value * weight + bias

# Mean squared error
cost = Tf::Math.reduce_sum((pred - y_value)**2.0) / (2.0 * train_y.length)

# Set up an optimizer
learning_rate = 0.05
optimizer = Tf::Train::GradientDescentOptimizer.new(learning_rate).minimize(cost)

# Create a new session and initialize variables
session = Tf::Graph::Session.new(Tf::Graph::Graph.default, Tf::Graph::SessionOptions.new)
session.run(Tf.global_variables_initializer)

# Setup a variable to keep track of the epoch and get an op to increment it
epoch_var = Tf::Variable.new(1, dtype: :int64)
session.run(epoch_var.initializer)
epoch_var_add_op = epoch_var.assign_add(1)

# Enable logging to Tensorbaord - create a file writer and initialize it
path = File.join(Dir.tmpdir, 'tensorflow-ruby')
Pathname(path).glob('*').each(&:delete)
writer = Tf::Summary.create_file_writer(path)
writer.step = epoch_var
writer_flush_op = writer.flush
session.run(writer.initializer)

# Log the graph
session.run(writer.graph(Tf::Graph::Graph.default))
session.run(writer_flush_op)

# Setup op to log cost
write_cost_op = writer.scalar("Cost", cost)

start_time = Time.now

# Train the data over 250 epochs
(0..250).each_with_index do |epoch, i|
  train_x.zip(train_y).each do |x, y|
    session.run(optimizer, {x_value => x, y_value => y})
  end

  current_cost = session.run(cost, {x_value => train_x, y_value => train_y})

  # Log the cost
  session.run(write_cost_op, {x_value => train_x, y_value => train_y})
  session.run(writer_flush_op)

  # Increment the epoch variable
  session.run(epoch_var_add_op)
  current_cost, current_weight, current_bias = session.run([cost, weight, bias], {x_value => train_x, y_value => train_y})
  STDOUT << "Epoch: " << epoch << ", cost= " << current_cost << ", W=" << current_weight << ", b=" << current_bias << "\n"
end

STDOUT << "\n"
STDOUT << "Optimization Finished!" << "\n"
STDOUT << "Time: " << (Time.now - start_time) << " seconds"
current_cost, current_weight, current_bias = session.run([cost, weight, bias], {x_value => train_x, y_value => train_y})
STDOUT << "Cost= " << current_cost << ", W=" << current_weight << ", b=" << current_bias << "\n"