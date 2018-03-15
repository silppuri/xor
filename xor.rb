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
  Layer.new(input_count: 3, output_count: 3, activation: SigmoidActivation),
  Layer.new(input_count: 3, output_count: 1, activation: SigmoidActivation)
)

network_input = [
  #bias, A, B
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

3001.times do |epoch|
  if (epoch % 100).zero?
    puts "---- epoch #{epoch} ----"
  end

  network_input.zip(expected_outputs) do |inputs, expected_output|
    actual_output = network.forward(inputs)
    errors = actual_output - expected_output

    if (epoch % 100).zero?
      puts " %d XOR %d = %.2f    error = %+.2f" % [
        inputs[1], inputs[2],
        actual_output[0],
        errors[0]
      ]
    end
    network.back_propagate errors
  end
end

puts "\n"
puts "*"*30
puts "TEST"
puts "*"*30
puts "0 XOR 0 = #{network.forward([1, 0, 0].to_vector).first.round}"
puts "0 XOR 1 = #{network.forward([1, 0, 1].to_vector).first.round}"
puts "1 XOR 0 = #{network.forward([1, 1, 0].to_vector).first.round}"
puts "1 XOR 1 = #{network.forward([1, 1, 1].to_vector).first.round}"
