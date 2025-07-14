#ifndef TRANSFORMERENCODERBLOCK_HPP
#define TRANSFORMERENCODERBLOCK_HPP

#include "layers/FeedForward.hpp"
#include "layers/Layer.hpp"
#include "layers/LayerNorm.hpp"
#include "layers/MultiHeadAttention.hpp"
#include <vector>

// Implementa un bloque codificador completo de un Transformer.
// Contiene dos sub-capas principales:
// 1. Multi-Head Self-Attention.
// 2. Red Feed-Forward (MLP).
// Cada sub-capa esta precedida por LayerNorm y seguida por una conexion residual.
class TransformerEncoderBlock : public Layer {
public:
  // Constructor del bloque codificador.
  TransformerEncoderBlock(size_t embedding_dim, size_t num_heads, size_t mlp_hidden_dim);

  // Realiza el paso hacia adelante a traves del bloque completo.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Realiza el paso hacia atras a traves del bloque completo.
  Tensor backward(const Tensor &outputGradient) override;

  // Recolecta los parametros de todas las sub-capas.
  std::vector<Tensor *> getParameters() override;

  // Recolecta los gradientes de todas las sub-capas.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "TransformerEncoderBlock"; }

private:
  // Componentes del bloque.
  LayerNorm norm1;
  MultiHeadAttention attention;
  LayerNorm norm2;
  FeedForward ffn;

  // Tensores guardados para las conexiones residuales en el backward pass.
  Tensor input_skip1; // Entrada a la primera conexion residual.
  Tensor input_skip2; // Entrada a la segunda conexion residual.
};

#endif // TRANSFORMERENCODERBLOCK_HPP
