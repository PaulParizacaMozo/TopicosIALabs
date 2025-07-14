#ifndef RELU_HPP
#define RELU_HPP

#include "layers/Layer.hpp"

// Implementa la funcion de activacion Rectified Linear Unit (ReLU).
// Realiza la operacion no lineal elemento a elemento: f(x) = max(0, x).
class ReLU : public Layer {
public:
  // Constructor.
  ReLU();

  // Aplica la funcion ReLU elemento a elemento.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Calcula el gradiente de la funcion ReLU.
  Tensor backward(const Tensor &outputGradient) override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "ReLU"; }

private:
  // Almacena la entrada para el calculo del gradiente en backward.
  Tensor inputTensor;
};

#endif // RELU_HPP
