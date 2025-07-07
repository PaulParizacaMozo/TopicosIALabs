#ifndef TANH_HPP
#define TANH_HPP

#include "layers/Layer.hpp"
#include <cmath> // Para tanh()

// Hyperbolic Tangent (Tanh) activation function.
// f(x) = tanh(x)
class Tanh : public Layer {
public:
  Tanh();

  Tensor forward(const Tensor &input, bool isTraining) override;
  Tensor backward(const Tensor &outputGradient) override;

  std::string getName() const override { return "Tanh"; }

private:
  // Similar a Sigmoid, guardamos la salida para calcular la derivada eficientemente.
  // f'(x) = 1 - tanh^2(x) = 1 - f(x)^2
  Tensor outputTensor;
};

#endif // TANH_HPP
