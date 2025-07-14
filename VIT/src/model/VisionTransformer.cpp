#include "model/VisionTransformer.hpp"

// Constructor que inicializa todas las capas del modelo.
VisionTransformer::VisionTransformer(const ViTConfig &config)
    : config(config),
      embeddings(config.image_size, config.image_size, config.patch_size, config.in_channels, config.embedding_dim),
      final_norm(config.embedding_dim), mlp_head(config.embedding_dim, config.num_classes) {

  // Crea la pila de bloques codificadores.
  for (size_t i = 0; i < config.num_layers; ++i) {
    encoder_blocks.emplace_back(config.embedding_dim, config.num_heads, config.mlp_hidden_dim);
  }
}

// Encadena el forward pass de todo el modelo.
Tensor VisionTransformer::forward(const Tensor &input, bool isTraining) {
  // 1. Capa de Embeddings (parcheo, CLS token, pos. encoding).
  Tensor x = embeddings.forward(input, isTraining);

  // 2. Pila de bloques codificadores del Transformer.
  for (auto &block : encoder_blocks) {
    x = block.forward(x, isTraining);
  }

  // 3. Normalizacion final.
  x = final_norm.forward(x, isTraining);

  if (isTraining) {
    // Guarda la salida normalizada para el backward pass.
    this->final_norm_output = x;
  }

  // 4. Extrae solo el token CLS (en la posicion 0) para la clasificacion.
  Tensor cls_token = x.slice(1, 0, 1).contiguous().reshape({input.getShape()[0], config.embedding_dim});

  // 5. Cabeza de clasificacion (MLP).
  return mlp_head.forward(cls_token, isTraining);
}

// Encadena el backward pass de todo el modelo en orden inverso.
Tensor VisionTransformer::backward(const Tensor &outputGradient) {
  // 1. Propaga hacia atras a traves de la cabeza de clasificacion.
  Tensor grad = mlp_head.backward(outputGradient);
  size_t batchSize = outputGradient.getShape()[0];

  // 2. El gradiente esta solo para el token CLS. Hay que "re-inyectarlo"
  // en una secuencia completa de gradientes (con ceros para los otros tokens).
  size_t num_tokens = 1 + (config.image_size / config.patch_size) * (config.image_size / config.patch_size);
  Tensor grad_seq({batchSize, num_tokens, config.embedding_dim});
  grad_seq.fill(0.0f);

  // Copia el gradiente del CLS a la posicion 0 de la secuencia.
  Tensor cls_grad_slice = grad_seq.slice(1, 0, 1);
  Tensor grad_reshaped = grad.reshape(cls_grad_slice.getShape());
  // (Esta parte podria optimizarse con una operacion de copia de slices).
#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batchSize; ++b) {
    for (size_t d = 0; d < config.embedding_dim; ++d) {
      cls_grad_slice(b, 0, d) = grad(b, d);
    }
  }

  // 3. Propaga a traves de la normalizacion final.
  grad = final_norm.backward(grad_seq);

  // 4. Propaga a traves de los bloques codificadores en orden inverso.
  for (int i = encoder_blocks.size() - 1; i >= 0; --i) {
    grad = encoder_blocks[i].backward(grad);
  }

  // 5. Propaga a traves de la capa de embeddings.
  grad = embeddings.backward(grad);

  // Devuelve el gradiente final (con respecto a la imagen), aunque no suele usarse.
  return grad;
}

// Recolecta los parametros de todas las capas del modelo.
std::vector<Tensor *> VisionTransformer::getParameters() {
  std::vector<Tensor *> params;
  auto emb_params = embeddings.getParameters();
  params.insert(params.end(), emb_params.begin(), emb_params.end());

  for (auto &block : encoder_blocks) {
    auto block_params = block.getParameters();
    params.insert(params.end(), block_params.begin(), block_params.end());
  }

  auto norm_params = final_norm.getParameters();
  params.insert(params.end(), norm_params.begin(), norm_params.end());

  auto head_params = mlp_head.getParameters();
  params.insert(params.end(), head_params.begin(), head_params.end());
  return params;
}

// Recolecta los gradientes de todas las capas del modelo.
std::vector<Tensor *> VisionTransformer::getGradients() {
  std::vector<Tensor *> grads;
  auto emb_grads = embeddings.getGradients();
  grads.insert(grads.end(), emb_grads.begin(), emb_grads.end());

  for (auto &block : encoder_blocks) {
    auto block_grads = block.getGradients();
    grads.insert(grads.end(), block_grads.begin(), block_grads.end());
  }

  auto norm_grads = final_norm.getGradients();
  grads.insert(grads.end(), norm_grads.begin(), norm_grads.end());

  auto head_grads = mlp_head.getGradients();
  grads.insert(grads.end(), head_grads.begin(), head_grads.end());
  return grads;
}
