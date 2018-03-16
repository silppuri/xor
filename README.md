# Simple overfitting neural network examples implemented in Ruby

## Run the examples

This has three different examples:

### 1. And-gate

Simple perceptron which computes the and gate logic using 
heaviside activation

```
$ ruby -I. perceptron.rb
```

### 2. XOR-gate

Simple two-layer neural network that computes the XOR gate
with Sigmoid activation as the nonlinear activation function.

```
$ ruby -I. xor.rb
```


### 3. "MNIST"

Simple example how characters can be represented using 
5 x 5 arrays of 0 and 1's and make a neural network to 
memoize the characters.


```
$ ruby -I. mnist.rb
```
