#ifndef FLATTEN_HPP
#define FLATTEN_HPP

#include "layers/Layer.hpp"

/**
 * @class Flatten
 * @brief Una capa que remodela un tensor de entrada a una forma 2D.
 *
 * Esta capa es fundamental para conectar capas que operan en datos espaciales
 * (como Conv2D o Pooling2D) con capas que operan en datos vectoriales (como Dense).
 *
 * Transforma una entrada de forma (Batch, Dim1, Dim2, ...) en una salida de
 * forma (Batch, Dim1 * Dim2 * ...), preservando la dimensión del batch.
 * No tiene parámetros entrenables.
 */
class Flatten : public Layer {
public:
  /** @brief Constructor por defecto. */
  Flatten();

  /**
   * @brief Aplana el tensor de entrada.
   * @details Almacena la forma de entrada para el paso hacia atrás y luego
   *          remodela la entrada a (batch_size, flattened_features).
   * @param input El tensor de entrada (ej. de forma {B, C, H, W}).
   * @param isTraining Ignorado, ya que el comportamiento es el mismo.
   * @return Un nuevo tensor aplanado de forma {B, C*H*W}.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief "Des-aplana" el gradiente de salida.
   * @details Utiliza la forma de entrada almacenada para remodelar el gradiente
   *          plano entrante a la forma original de la entrada de la capa.
   * @param outputGradient Gradiente de forma {B, flattened_features}.
   * @return Un gradiente remodelado a la forma de entrada original (ej. {B, C, H, W}).
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Devuelve el nombre de la capa.
   * @return El string "Flatten".
   * @override
   */
  std::string getName() const override { return "Flatten"; }

private:
  /** @brief Almacena la forma de la entrada durante el forward pass.
   *         Es necesario para poder reconstruir la forma en el backward pass.
   */
  std::vector<size_t> inputShape;
};

#endif // FLATTEN_HPP
