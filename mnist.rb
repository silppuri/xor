require 'pry'
require 'chars'
require 'neural_network'

# MNIST example

network = NeuralNetwork.new(
  Layer.new(input_count: 25, output_count: 10, activation: SigmoidActivation),
  Layer.new(input_count: 10, output_count: 4, activation: SigmoidActivation),
)


network_input = [
  Chars::A[:value].flatten.to_vector,
  Chars::B[:value].flatten.to_vector,
  Chars::C[:value].flatten.to_vector,
  Chars::D[:value].flatten.to_vector,
]

expected_outputs = [
  Chars::A[:expected].flatten.to_vector,
  Chars::B[:expected].flatten.to_vector,
  Chars::C[:expected].flatten.to_vector,
  Chars::D[:expected].flatten.to_vector,
]

def map_index_to_char(i)
  ('A'..'D').to_a[i]
end

def softmax(mat)
  mat.map!{|el| Math::exp(el) }
  sum = mat.inject(0){|sum,el| sum = sum + el}
  mat.map{|el| el / sum.to_f}
end

1200.times do |epoch|
  if (epoch % 100).zero?
    puts "---- epoch #{epoch} ----"
  end

  network_input.zip(expected_outputs) do |inputs, expected_output|
    actual_output = network.forward(inputs)
    errors = actual_output - expected_output

    if (epoch % 100).zero?
        predictions = softmax(actual_output.to_a)
        expected = softmax(expected_output.to_a)
        puts "given input     #{inputs}"
        puts "expected output #{expected_output}"
        puts "actual output   #{predictions}"
        puts "errors          #{errors}"
        puts "expected char   #{map_index_to_char(expected.rindex(expected.max))}"
        puts "actual char     #{map_index_to_char(predictions.rindex(predictions.max))}"
    end

    network.back_propagate errors
  end
end

puts "\n"
puts "*"*30
puts "D-like prediction"
puts "*"*30
puts softmax(network.forward(
  [
    [0,1,1,1,0],
    [0,1,0,0,1],
    [0,1,0,0,1],
    [0,1,0,0,1],
    [0,1,1,1,0]
  ].flatten.to_vector).to_a)
