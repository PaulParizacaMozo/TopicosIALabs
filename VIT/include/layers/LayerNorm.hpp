#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP

#include "layers/Layer.hpp"

// Implementa la Normalizacion de Capa (Layer Normalization).
// Normaliza las activaciones a lo largo de la dimension de caracteristicas.
// y = gamma * (x - mean) / sqrt(var + epsilon) + beta
class LayerNorm : public Layer {
public:
  // Constructor. Define el tamaño de la dimension a normalizar.
  LayerNorm(size_t featureSize, float epsilon = 1e-5f);

  // Realiza el paso de normalizacion hacia adelante.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Calcula los gradientes para gamma, beta y la entrada.
  Tensor backward(const Tensor &outputGradient) override;

  // Devuelve los parametros entrenables: gamma y beta.
  std::vector<Tensor *> getParameters() override;

  // Devuelve los gradientes de los parametros.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "LayerNorm"; }

private:
  float epsilon;
  size_t featureSize;

  // Parametros entrenables
  Tensor gamma; // Parametro de escala, forma {1, 1, ..., feature_size}.
  Tensor beta;  // Parametro de desplazamiento, forma {1, 1, ..., feature_size}.

  // Gradientes de los parametros
  Tensor gammaGradient;
  Tensor betaGradient;

  // Estado para el backward pass
  Tensor inputTensor;     // Copia de la entrada del forward.
  Tensor mean;            // Media por cada muestra.
  Tensor variance;        // Varianza por cada muestra.
  Tensor normalizedInput; // Entrada normalizada antes de gamma/beta.
};

#endif // LAYERNOWN_HPP
