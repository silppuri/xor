require 'pry'
require 'matrix'

class Array
  def to_vector
    Vector.elements self
  end

  def sum
    inject(:+)
  end
end

module Math
  def self.sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
  end

  def self.sigmoid_derivate(x)
    x * (1 - x)
  end
end

class NeuralNetwork
  def initialize(*layers)
    @layers = layers
  end

  def forward(inputs)
    outputs = nil
    @layers.each do |layer|
      outputs = layer.forward(inputs)
      inputs = outputs
    end
    outputs
  end

  def back_propagate(errors)
    @layers.reverse_each do |layer|
      errors = layer.back_propagate(errors)
    end
  end

  def update_weights
    @layers.each(&:update_weights)
  end
end

class Layer
  def initialize(input_count, output_count)
    @neurons = Array.new(output_count) { Neuron.new(input_count) }
  end

  def forward(inputs)
    @neurons.map { |neuron| neuron.forward(inputs) }.to_vector
  end

  def back_propagate(errors)
    @neurons.zip(errors).map { |neuron, error| neuron.back_propagate(error) }.sum
  end

  def update_weights
    @neurons.each(&:update_weights)
  end
end

class Neuron
  def initialize(inputs_count)
    @weights = Array.new(inputs_count) { rand }.to_vector
  end

  def forward(inputs)
    @inputs = inputs
    @output = Math.sigmoid(inputs.dot(@weights))
  end

  def back_propagate(error)
    @delta = error * Math.sigmoid_derivate(@output)
    @weights * @delta
  end

  def update_weights
    @weights -= @inputs * @delta
  end
end
