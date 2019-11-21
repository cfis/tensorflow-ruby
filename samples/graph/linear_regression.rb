$LOAD_PATH.unshift(File.expand_path('../../lib', __dir__))

require "tensorflow"
Tensorflow.disable_eager_execution

learning_rate = 0.01
training_epochs = 1000
display_step = 50

train_x = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1,]

train_y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3,]

n_samples = train_x.size

x_value = Tensorflow.placeholder(:float, name: 'x_value')
y_value = Tensorflow.placeholder(:float, name: 'y_value')

# Set model weights
weight = Tensorflow::Variable.new(rand, name: 'weight', trainable: true)
bias = Tensorflow::Variable.new(rand, name: 'bias', trainable: true)

# Construct a linear model
pred = x_value * weight + bias

# Mean squared error
cost = Tensorflow::Math.reduce_sum((pred - y_value)**2.0) / (2.0 * n_samples)

optimizer = Tensorflow::Train::GradientDescentOptimizer.new(learning_rate).minimize(cost)

session = Tensorflow::Graph::Session.new(Tensorflow::Graph::Graph.default, Tensorflow::Graph::SessionOptions.new)
session.run(Tensorflow.global_variables_initializer)

result = session.run([cost], x_value => 5.0, y_value => 10.0)
puts result

start_time = Time.now

(0..training_epochs).each do |epoch|
  train_x.zip(train_y).each do |x, y|
    session.run(optimizer, {x_value => x, y_value => y})
  end

  if (epoch + 1) % display_step == 0
    current_cost = session.run(cost, {x_value => train_x, y_value => train_y})
    current_weight = session.run(weight)
    current_bias = session.run(bias)

    puts("Epoch:", "%04d" % (epoch + 1), "cost=", current_cost, "W=", current_weight, "b=", current_bias)
  end
end

puts "Optimization Finished!"
training_cost = session.run(cost, {x_value => train_x, y_value => train_y})
puts "Training cost=", training_cost, "W=", session.run(weight), "b=", session.run(bias), '\n'
puts "time elapsed ", Time.now.to_i - start_time.to_i
