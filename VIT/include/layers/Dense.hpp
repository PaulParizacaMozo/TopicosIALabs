#ifndef DENSE_HPP
#define DENSE_HPP

#include "layers/Layer.hpp"
#include <cmath>

// Capa totalmente conectada (fully connected).
// Realiza la operacion: output = input * weights + bias.
class Dense : public Layer {
public:
  // Constructor. Define las dimensiones de entrada y salida.
  Dense(size_t inputSize, size_t outputSize);

  // Realiza la transformacion afin: Y = X * W + b.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Calcula los gradientes para los pesos, el bias y la entrada.
  Tensor backward(const Tensor &outputGradient) override;

  // Devuelve los parametros entrenables: pesos y bias.
  std::vector<Tensor *> getParameters() override;

  // Devuelve los gradientes de los parametros.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "Dense"; }

private:
  // Parametros entrenables
  Tensor weights; // Matriz de pesos, forma {input_size, output_size}.
  Tensor bias;    // Vector de bias, forma {1, output_size}.

  // Gradientes de los parametros
  Tensor weightGradients; // Gradiente de los pesos.
  Tensor biasGradients;   // Gradiente del bias.

  // Almacena la entrada del forward pass para el calculo del backward pass.
  Tensor inputTensor;
};

#endif // DENSE_HPP
