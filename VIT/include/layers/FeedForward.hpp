#ifndef FEEDFORWARD_HPP
#define FEEDFORWARD_HPP

#include "activations/GELU.hpp"
#include "layers/Dense.hpp"
#include "layers/Layer.hpp"
#include <vector>

// Implementa la red Feed-Forward (MLP) del bloque Transformer.
// Consiste en dos capas lineales con una activacion no lineal en medio:
// Dense -> GELU -> Dense.
class FeedForward : public Layer {
public:
  // Constructor. Define la dimension de entrada/salida y la dimension oculta.
  FeedForward(size_t embedding_dim, size_t hidden_dim);

  // Realiza el paso hacia adelante a traves de las capas internas.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Realiza el paso hacia atras en orden inverso al forward.
  Tensor backward(const Tensor &outputGradient) override;

  // Recolecta los parametros de las capas Dense internas.
  std::vector<Tensor *> getParameters() override;

  // Recolecta los gradientes de las capas Dense internas.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "FeedForward"; }

private:
  // Capas que componen la red Feed-Forward.
  Dense dense1;
  GELU activation;
  Dense dense2;
};

#endif // FEEDFORWARD_HPP
