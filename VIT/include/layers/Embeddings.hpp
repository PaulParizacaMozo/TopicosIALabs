#ifndef EMBEDDINGS_HPP
#define EMBEDDINGS_HPP

#include "layers/Layer.hpp"
#include "layers/PatchEmbedding.hpp"
#include <memory>

// Encapsula toda la logica de preparacion de la entrada para un ViT.
// Realiza tres operaciones clave:
// 1. Usa PatchEmbedding para convertir imagenes en embeddings de parches.
// 2. Pre-añade un token de clasificacion [CLS] entrenable a la secuencia.
// 3. Suma una codificacion posicional entrenable a la secuencia combinada.
class Embeddings : public Layer {
public:
  // Constructor.
  Embeddings(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim);

  // Realiza el paso hacia adelante.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Realiza el paso hacia atras.
  Tensor backward(const Tensor &outputGradient) override;

  // Recolecta los parametros de PatchEmbedding y los propios (CLS, Positional).
  std::vector<Tensor *> getParameters() override;

  // Recolecta los gradientes de PatchEmbedding y los propios.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "Embeddings"; }

private:
  // Capa de parcheo contenida.
  std::unique_ptr<PatchEmbedding> patcher;

  // Parametros entrenables propios de esta capa
  Tensor clsToken;           // Forma {1, 1, embedding_dim}.
  Tensor positionalEncoding; // Forma {1, num_patches + 1, embedding_dim}.

  // Gradientes correspondientes
  Tensor clsTokenGradient;
  Tensor positionalEncodingGradient;

  // Dimensiones guardadas
  size_t num_patches;
  size_t embedding_dim;
};

#endif // EMBEDDINGS_HPP
