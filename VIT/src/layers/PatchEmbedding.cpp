#include "layers/PatchEmbedding.hpp"
#include <stdexcept>

PatchEmbedding::PatchEmbedding(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels,
                               size_t embedding_dim)
    : image_height(image_height), image_width(image_width), patch_size(patch_size), in_channels(in_channels),
      embedding_dim(embedding_dim) {

  if (image_height % patch_size != 0 || image_width % patch_size != 0) {
    throw std::invalid_argument("Las dimensiones de la imagen deben ser divisibles por el tamaño del parche.");
  }

  size_t num_patches_h = image_height / patch_size;
  size_t num_patches_w = image_width / patch_size;
  this->num_patches = num_patches_h * num_patches_w;
  this->patch_dim = patch_size * patch_size * in_channels;

  // Inicializa la capa de proyeccion lineal.
  this->projectionLayer = std::make_unique<Dense>(this->patch_dim, this->embedding_dim);
}

Tensor PatchEmbedding::forward(const Tensor &input, bool isTraining) {
  const auto &inputShape = input.getShape();
  size_t batchSize = inputShape[0];

  // Tensor para almacenar los parches aplanados, listo para la capa Densa.
  Tensor patches_flat({batchSize * this->num_patches, this->patch_dim});

  size_t patch_index_global = 0;
  // Itera sobre cada imagen en el batch.
  for (size_t b = 0; b < batchSize; ++b) {
    // Itera para extraer cada parche.
    for (size_t ph = 0; ph < image_height / patch_size; ++ph) {
      for (size_t pw = 0; pw < image_width / patch_size; ++pw) {
        // Extrae el parche como una vista, sin copiar datos.
        size_t h_start = ph * patch_size;
        size_t w_start = pw * patch_size;
        Tensor patch_view = input
                                .slice(0, b, 1)                 // Vista de la imagen b
                                .slice(2, h_start, patch_size)  // Vista de la fila de parches
                                .slice(3, w_start, patch_size); // Vista del parche exacto

        // Copia la vista del parche aplanado a nuestro tensor de parches.
        // Asumimos que la vista del parche es contigua.
        const float *patch_data = patch_view.getData() + patch_view.getDataOffset();
        float *dest_data = patches_flat.getData() + (patch_index_global * this->patch_dim);
        std::copy(patch_data, patch_data + this->patch_dim, dest_data);

        patch_index_global++;
      }
    }
  }

  if (isTraining) {
    this->flattenedPatches = patches_flat;
  }

  // Proyecta los parches aplanados al espacio de embedding.
  Tensor projected_patches = this->projectionLayer->forward(patches_flat, isTraining);
  return projected_patches.reshape({batchSize, this->num_patches, this->embedding_dim});
}

Tensor PatchEmbedding::backward(const Tensor &outputGradient) {
  const auto &gradShape = outputGradient.getShape();
  size_t batchSize = gradShape[0];

  // 1. Propagar el gradiente hacia atras a traves de la capa de proyeccion.
  Tensor grad2D = outputGradient.reshape({batchSize * this->num_patches, this->embedding_dim});
  Tensor patch_gradient = this->projectionLayer->backward(grad2D); // -> {B*num_patches, patch_dim}

  // 2. "Des-parchear" el gradiente, escribiendolo de vuelta en la forma de la imagen.
  Tensor input_gradient({batchSize, this->in_channels, this->image_height, this->image_width});
  input_gradient.fill(0.0f);

  size_t patch_index_global = 0;
  for (size_t b = 0; b < batchSize; ++b) {
    for (size_t ph = 0; ph < image_height / patch_size; ++ph) {
      for (size_t pw = 0; pw < image_width / patch_size; ++pw) {
        // Vista del gradiente del parche actual.
        Tensor current_patch_grad = patch_gradient.slice(0, patch_index_global, 1);
        const float *grad_data = current_patch_grad.getData();

        // Escribe el gradiente del parche de vuelta en su posicion original.
        for (size_t c = 0; c < this->in_channels; ++c) {
          for (size_t h = 0; h < this->patch_size; ++h) {
            for (size_t w = 0; w < this->patch_size; ++w) {
              size_t grad_idx = c * (this->patch_size * this->patch_size) + h * this->patch_size + w;
              input_gradient(b, c, ph * this->patch_size + h, pw * this->patch_size + w) = grad_data[grad_idx];
            }
          }
        }
        patch_index_global++;
      }
    }
  }
  return input_gradient;
}

std::vector<Tensor *> PatchEmbedding::getParameters() { return this->projectionLayer->getParameters(); }

std::vector<Tensor *> PatchEmbedding::getGradients() { return this->projectionLayer->getGradients(); }
