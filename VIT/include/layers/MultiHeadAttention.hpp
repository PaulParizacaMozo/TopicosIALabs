#ifndef MULTIHEADATTENTION_HPP
#define MULTIHEADATTENTION_HPP

#include "layers/Dense.hpp"
#include "layers/Layer.hpp"
#include <memory>
#include <vector>

// Implementa el mecanismo de Atencion Multi-Cabeza (Multi-Head Attention).
// Es el componente central de los bloques Transformer.
class MultiHeadAttention : public Layer {
public:
  // Constructor.
  // - embedding_dim: Dimension de los embeddings de entrada y salida (D).
  // - num_heads: Numero de cabezas de atencion (h). Debe dividir a D.
  MultiHeadAttention(size_t embedding_dim, size_t num_heads);

  // Realiza el paso hacia adelante de la atencion.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Realiza el paso hacia atras.
  Tensor backward(const Tensor &outputGradient) override;

  // Recolecta los parametros de las capas Dense internas (Q, K, V, Out).
  std::vector<Tensor *> getParameters() override;

  // Recolecta los gradientes de las capas Dense internas.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "MultiHeadAttention"; }

private:
  size_t embedding_dim;
  size_t num_heads;
  size_t head_dim; // Dimension de cada cabeza (D / h).

  // Capas de proyeccion lineal para Query, Key, Value y la salida.
  std::unique_ptr<Dense> q_proj;
  std::unique_ptr<Dense> k_proj;
  std::unique_ptr<Dense> v_proj;
  std::unique_ptr<Dense> out_proj;

  // Funcion auxiliar para la atencion escalada por producto punto.
  Tensor scaledDotProductAttention(const Tensor &q, const Tensor &k, const Tensor &v);

  // Tensores guardados para el backward pass.
  Tensor inputTensor;               // Entrada original.
  Tensor q_split, k_split, v_split; // Proyecciones Q, K, V divididas por cabeza.
  Tensor attention_weights;         // Pesos de atencion despues de softmax.
};

#endif // MULTIHEADATTENTION_HPP
