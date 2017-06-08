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

# 
# The netowrk has shape of
#
#  x1 - O
#    \ / \
#     X   O -> y
#    / \ /
#  x2 - O

INPUTS_COUNT = 3
HIDDEN_COUNT = 3
OUTPUT_COUNT = 1

network = NeuralNetwork.new(
  Layer.new(INPUTS_COUNT, HIDDEN_COUNT),
  Layer.new(HIDDEN_COUNT, OUTPUT_COUNT)
)

examples = [
  [1, 0, 0].to_vector,
  [1, 0, 1].to_vector,
  [1, 1, 0].to_vector,
  [1, 1, 1].to_vector
]

expected_outputs = [
  [0].to_vector,
  [1].to_vector,
  [1].to_vector,
  [0].to_vector
]

5001.times do |epoch|
  if (epoch % 100).zero?
    puts "-- epoch #{epoch} --"
  end
  examples.zip(expected_outputs) do |inputs, expected_output|
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
    network.update_weights
  end
end
