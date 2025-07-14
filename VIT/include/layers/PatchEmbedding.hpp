#ifndef PATCHEMBEDDING_HPP
#define PATCHEMBEDDING_HPP

#include "layers/Dense.hpp"
#include "layers/Layer.hpp"
#include <memory>

// Convierte un lote de imagenes en una secuencia de embeddings de parches.
// Realiza dos pasos:
// 1. Divide las imagenes en parches fijos.
// 2. Aplana cada parche y lo proyecta a la dimension del embedding
//    a traves de una capa lineal (Dense).
class PatchEmbedding : public Layer {
public:
  // Constructor.
  PatchEmbedding(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim);

  // Realiza el paso de parcheo y proyeccion.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Realiza el paso hacia atras a traves de la proyeccion y el "des-parcheo".
  Tensor backward(const Tensor &outputGradient) override;

  // Devuelve los parametros de la capa de proyeccion interna.
  std::vector<Tensor *> getParameters() override;

  // Devuelve los gradientes de la capa de proyeccion interna.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "PatchEmbedding"; }

  // Devuelve el numero de parches generados.
  size_t getNumPatches() const { return num_patches; }

private:
  size_t image_height, image_width, patch_size, in_channels, embedding_dim;
  size_t patch_dim;   // Dimension del parche aplanado (patch_size * patch_size * channels).
  size_t num_patches; // Numero total de parches por imagen.

  // Capa de proyeccion lineal.
  std::unique_ptr<Dense> projectionLayer;

  // Tensor con los parches aplanados, guardado para el backward pass.
  Tensor flattenedPatches;
};

#endif // PATCHEMBEDDING_HPP
