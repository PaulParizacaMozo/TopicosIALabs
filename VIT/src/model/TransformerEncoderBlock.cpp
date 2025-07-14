#include "model/TransformerEncoderBlock.hpp"

// Constructor que inicializa todas las sub-capas del bloque.
TransformerEncoderBlock::TransformerEncoderBlock(size_t embedding_dim, size_t num_heads, size_t mlp_hidden_dim)
    : norm1(embedding_dim), attention(embedding_dim, num_heads), norm2(embedding_dim), ffn(embedding_dim, mlp_hidden_dim) {}

// Define el flujo de datos forward del bloque, incluyendo las conexiones residuales.
Tensor TransformerEncoderBlock::forward(const Tensor &input, bool isTraining) {
  if (isTraining) {
    // Guarda la entrada para la primera conexion residual en backward.
    this->input_skip1 = input;
  }

  // Sub-capa 1: Multi-Head Attention (Pre-LN).
  // output = input + Attention(LayerNorm(input))
  Tensor x = norm1.forward(input, isTraining);
  x = attention.forward(x, isTraining);
  Tensor residual1 = input + x;

  if (isTraining) {
    // Guarda la entrada de la segunda conexion residual.
    this->input_skip2 = residual1;
  }

  // Sub-capa 2: Feed-Forward Network (Pre-LN).
  // output = residual1 + FFN(LayerNorm(residual1))
  Tensor y = norm2.forward(residual1, isTraining);
  y = ffn.forward(y, isTraining);
  return residual1 + y;
}

// Define el flujo de gradientes hacia atras, manejando las conexiones residuales.
Tensor TransformerEncoderBlock::backward(const Tensor &outputGradient) {
  // Para una conexion residual Y = X + F(X), el gradiente de X es dL/dY + dL/d(F(X)).
  // El gradiente de la salida (dL/dY) se propaga por ambas ramas.

  // --- Inversa de la segunda conexion residual ---
  Tensor grad_skip2 = outputGradient;
  Tensor grad_ffn = outputGradient; // El mismo gradiente entra en la rama FFN.

  grad_ffn = ffn.backward(grad_ffn);
  grad_ffn = norm2.backward(grad_ffn);

  // Suma de los gradientes de la rama skip y la rama FFN.
  Tensor grad_residual1 = grad_skip2 + grad_ffn;

  // --- Inversa de la primera conexion residual ---
  Tensor grad_skip1 = grad_residual1;
  Tensor grad_mha = grad_residual1; // El mismo gradiente entra en la rama de atencion.

  grad_mha = attention.backward(grad_mha);
  grad_mha = norm1.backward(grad_mha);

  // Suma de los gradientes para obtener el gradiente final de la entrada.
  return grad_skip1 + grad_mha;
}

// Recolecta los parametros de todas las sub-capas.
std::vector<Tensor *> TransformerEncoderBlock::getParameters() {
  std::vector<Tensor *> params;
  auto p1 = norm1.getParameters();
  params.insert(params.end(), p1.begin(), p1.end());
  auto p2 = attention.getParameters();
  params.insert(params.end(), p2.begin(), p2.end());
  auto p3 = norm2.getParameters();
  params.insert(params.end(), p3.begin(), p3.end());
  auto p4 = ffn.getParameters();
  params.insert(params.end(), p4.begin(), p4.end());
  return params;
}

// Recolecta los gradientes de todas las sub-capas.
std::vector<Tensor *> TransformerEncoderBlock::getGradients() {
  std::vector<Tensor *> grads;
  auto g1 = norm1.getGradients();
  grads.insert(grads.end(), g1.begin(), g1.end());
  auto g2 = attention.getGradients();
  grads.insert(grads.end(), g2.begin(), g2.end());
  auto g3 = norm2.getGradients();
  grads.insert(grads.end(), g3.begin(), g3.end());
  auto g4 = ffn.getGradients();
  grads.insert(grads.end(), g4.begin(), g4.end());
  return grads;
}
