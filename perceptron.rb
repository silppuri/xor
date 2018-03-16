require 'neural_network'

# Implementation of AND gate with
# perceptron.
#
#  A | B | A AND B
#  ---------------
#  0 | 0 |   0
#  0 | 1 |   0
#  1 | 0 |   0
#  1 | 1 |   1
#
#   B
#   \
#   |\
#   | \
# 1 0  \ 1
#   |   \
#   |    \
#  -0----0\--- A
#   |    1 \

perceptron = NeuralNetwork.new(
  Layer.new(input_count: 3, output_count: 1, activation: HeavisideActivation)
)

and_inputs = [
  #bias, A, B
  [1, 0, 0].to_vector,
  [1, 0, 1].to_vector,
  [1, 1, 0].to_vector,
  [1, 1, 1].to_vector
]

expected_outputs = [
  [0].to_vector,
  [0].to_vector,
  [0].to_vector,
  [1].to_vector
]

10.times do |epoch|
  puts "---- epoch #{epoch} ----"
  and_inputs.zip(expected_outputs) do |inputs, expected_output|
    actual_output = perceptron.forward(inputs)
    errors = actual_output - expected_output

    puts " %d AND %d = %.2f    error = %+.2f" % [
      inputs[1], inputs[2],
      actual_output[0],
      errors[0]
    ]

    perceptron.layers.flat_map(&:neurons).each do |neuron|
      new_weights = neuron.weights.each_with_index.map do |_, i|
        0.1 * errors.first * inputs[i]
      end.to_vector
      neuron.weights -= new_weights
    end
  end
end
