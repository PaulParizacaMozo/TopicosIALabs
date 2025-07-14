#ifndef GELU_HPP
#define GELU_HPP

#include "layers/Layer.hpp"

// Implementa la funcion de activacion GELU (Gaussian Error Linear Unit).
// Usa una aproximacion rapida popular en modelos Transformer.
class GELU : public Layer {
public:
  // Constructor.
  GELU();

  // Aplica la funcion de activacion GELU elemento por elemento.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Calcula el gradiente de la funcion GELU.
  Tensor backward(const Tensor &outputGradient) override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "GELU"; }

private:
  // Almacena la entrada para el calculo del gradiente en backward.
  Tensor inputTensor;
};

#endif // GELU_HPP
