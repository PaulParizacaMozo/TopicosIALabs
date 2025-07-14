#ifndef VISIONTRANSFORMER_HPP
#define VISIONTRANSFORMER_HPP

#include "layers/Dense.hpp"
#include "layers/Embeddings.hpp"
#include "layers/Layer.hpp"
#include "layers/LayerNorm.hpp"
#include "model/TransformerEncoderBlock.hpp"
#include <memory>
#include <vector>

// Estructura para mantener todos los hiperparametros del Vision Transformer.
struct ViTConfig {
  size_t image_size = 28;
  size_t patch_size = 7;
  size_t in_channels = 1;
  size_t num_classes = 10;
  size_t embedding_dim = 128;
  size_t num_heads = 8;
  size_t num_layers = 4;
  size_t mlp_hidden_dim = 512;
};

// Implementacion completa del modelo Vision Transformer.
// Encapsula la capa de embeddings, la pila de bloques codificadores y la
// cabeza de clasificacion (MLP Head) en una unica interfaz de Layer.
class VisionTransformer : public Layer {
public:
  // Constructor que construye el modelo a partir de una configuracion.
  explicit VisionTransformer(const ViTConfig &config);

  // Realiza un forward pass completo a traves de todo el modelo.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Realiza un backward pass completo a traves de todo el modelo.
  Tensor backward(const Tensor &outputGradient) override;

  // Recolecta los parametros de todas las capas del modelo.
  std::vector<Tensor *> getParameters() override;

  // Recolecta los gradientes de todas las capas del modelo.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre del modelo.
  std::string getName() const override { return "VisionTransformer"; }

private:
  ViTConfig config;

  // Las partes del modelo.
  Embeddings embeddings;
  std::vector<TransformerEncoderBlock> encoder_blocks;
  LayerNorm final_norm;
  Dense mlp_head;

  // Tensor guardado para el backward pass.
  Tensor final_norm_output;
};

#endif // VISIONTRANSFORMER_HPP
