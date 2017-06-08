require 'pry'
require 'neural_network'

#
# Implementation of XOR gate with
# simple neural network
#
#  A | B | A XOR B
#  ---------------
#  0 | 0 |   0
#  0 | 1 |   1
#  1 | 0 |   1
#  1 | 1 |   0
#

network = NeuralNetwork.new(
  Layer.new(input_count: 3, output_count: 3),
  Layer.new(input_count: 3, output_count: 1)
)

network_input = [
  #bias, A, B
  [rand, 0, 0].to_vector,
  [rand, 0, 1].to_vector,
  [rand, 1, 0].to_vector,
  [rand, 1, 1].to_vector
]

expected_outputs = [
  [0].to_vector,
  [1].to_vector,
  [1].to_vector,
  [0].to_vector
]

3001.times do |epoch|
  if (epoch % 100).zero?
    puts "---- epoch #{epoch} ----"
  end

  network_input.zip(expected_outputs) do |inputs, expected_output|
    actual_output = network.forward(inputs)
    errors = actual_output - expected_output

    if (epoch % 100).zero?
      puts " %d XOR %d = %.2f    error = %+.2f" % [
        inputs[0], inputs[1],
        actual_output[0],
        errors[0]
      ]
    end
    network.back_propagate errors
  end
end
