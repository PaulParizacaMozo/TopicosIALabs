#include "layers/Embeddings.hpp"
#include "core/Tensor.hpp"

Embeddings::Embeddings(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim)
    : embedding_dim(embedding_dim) {

  // Inicializa la capa de parcheo interna.
  patcher = std::make_unique<PatchEmbedding>(image_height, image_width, patch_size, in_channels, embedding_dim);
  this->num_patches = patcher->getNumPatches();

  // Inicializa los parametros entrenables con valores pequeños aleatorios.
  float stddev = 0.02f;
  clsToken = Tensor({1, 1, embedding_dim});
  clsToken.randomizeNormal(0.0f, stddev);

  positionalEncoding = Tensor({1, 1 + this->num_patches, embedding_dim});
  positionalEncoding.randomizeNormal(0.0f, stddev);

  // Inicializa los gradientes con la misma forma, a cero.
  clsTokenGradient = Tensor(clsToken.getShape());
  positionalEncodingGradient = Tensor(positionalEncoding.getShape());
}

Tensor Embeddings::forward(const Tensor &input, bool isTraining) {
  size_t batchSize = input.getShape()[0];

  // 1. Obtener los embeddings de los parches
  Tensor patch_embeddings = this->patcher->forward(input, isTraining); // -> {B, N, D}

  // 2. Expandir el token CLS para que coincida con el tamaño del batch
  // expand() crea una vista {B, 1, D} sin copiar memoria.
  Tensor cls_token_batch({batchSize, 1, this->embedding_dim});

  // 3. Concatenar el CLS token y los parches a lo largo del eje de la secuencia (axis=1)
  Tensor embeddings_with_cls = concatenate({cls_token_batch, patch_embeddings}, 1); // -> {B, N+1, D}

  // 4. Añadir la codificación posicional
  // addBroadcast suma {1, N+1, D} a cada muestra de {B, N+1, D}
  embeddings_with_cls.addBroadcast(this->positionalEncoding);

  return embeddings_with_cls;
}

Tensor Embeddings::backward(const Tensor &outputGradient) {
  // El gradiente de una suma es el mismo para ambas ramas.
  // Por tanto, el gradiente de la codificacion posicional es la suma a traves del batch.
  this->positionalEncodingGradient = outputGradient.sum(0); // -> {1, N+1, D}
  // El gradiente que fluye hacia la concatenacion es el mismo outputGradient.
  Tensor grad_before_pos = outputGradient;

  // "Des-concatenar" el gradiente obteniendo vistas (slices).
  Tensor grad_cls = grad_before_pos.slice(1, 0, 1);                          // -> {B, 1, D}
  Tensor grad_patches_view = grad_before_pos.slice(1, 1, this->num_patches); // -> {B, N, D}

  // El gradiente del token CLS es la suma a traves del batch de su gradiente.
  this->clsTokenGradient = grad_cls.sum(0); // -> {1, 1, D}

  // En lugar de llamar a .contiguous(), creamos un nuevo tensor y copiamos los datos.
  // Esto garantiza que el tensor que pasamos es 100% contiguo.
  Tensor grad_patches_contiguous(grad_patches_view.getShape());

  // Usamos el operator() que sabe cómo manejar los strides de la vista
  const auto &shape = grad_patches_view.getShape();
#pragma omp parallel for collapse(3)
  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      for (size_t k = 0; k < shape[2]; ++k) {
        grad_patches_contiguous(i, j, k) = grad_patches_view(i, j, k);
      }
    }
  }

  // Ahora pasamos este tensor garantizado-contiguo
  Tensor input_image_gradient = this->patcher->backward(grad_patches_contiguous);

  return input_image_gradient;
}

std::vector<Tensor *> Embeddings::getParameters() {
  // Obtenemos los parámetros de la capa interna (pesos y bias de la Dense)
  auto params = this->patcher->getParameters();
  // Añadimos nuestros propios parámetros
  params.push_back(&this->clsToken);
  params.push_back(&this->positionalEncoding);
  return params;
}

std::vector<Tensor *> Embeddings::getGradients() {
  auto grads = this->patcher->getGradients();
  grads.push_back(&this->clsTokenGradient);
  grads.push_back(&this->positionalEncodingGradient);
  return grads;
}
