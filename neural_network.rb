require 'matrix'

class Array
  def to_vector
    Vector.elements self
  end

  def sum
    inject(:+)
  end
end

module SigmoidActivation
  def self.activate(x)
    1.0 / (1.0 + Math.exp(-x))
  end

  def self.derivate(x)
    x * (1 - x)
  end
end

module HeavisideActivation
  def self.activate(x)
    x < 0 ? 0 : 1
  end

  def self.derivate(x)
    raise 'Heaviside\'s derivate is zero everywhere :('
  end
end

class NeuralNetwork
  attr_accessor :layers

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
end

class Layer
  attr_accessor :neurons

  def initialize(input_count:, output_count:, activation:)
    @neurons = Array.new(output_count) { Neuron.new(input_count, activation) }
  end

  def forward(inputs)
    @neurons.map { |neuron| neuron.forward(inputs) }.to_vector
  end

  def back_propagate(errors)
    @neurons.zip(errors).map { |neuron, error| neuron.back_propagate(error) }.sum
  end
end

class Neuron
  attr_accessor :weights

  def initialize(inputs_count, activation)
    @weights = Array.new(inputs_count) { rand }.to_vector
    @activation = activation
  end

  def forward(inputs)
    @inputs = inputs
    @output = @activation.activate(inputs.dot(@weights))
  end

  def back_propagate(error)
    @delta = error * @activation.derivate(@output)
    update_weights
    pass_error
  end

  private

    def update_weights
      @weights -= @inputs * @delta
    end

    def pass_error
      @weights * @delta
    end
end
