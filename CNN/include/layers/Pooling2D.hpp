#ifndef POOLING2D_HPP
#define POOLING2D_HPP

#include "layers/Layer.hpp"
#include <string>

/**
 * @class Pooling2D
 * @brief Una capa que realiza submuestreo espacial (downsampling) en los mapas de características.
 *
 * El pooling reduce las dimensiones de altura y anchura de la entrada. Esto tiene
 * dos beneficios principales:
 * 1. Reduce la cantidad de parámetros y cómputo en la red.
 * 2. Ayuda a que las características detectadas sean más invariantes a pequeñas
 *    traslaciones en la entrada (robustez posicional).
 *
 * Soporta dos tipos comunes de pooling: Max Pooling y Average Pooling.
 */
class Pooling2D : public Layer {
public:
  /** @brief Define el tipo de operación de pooling a realizar. */
  enum class PoolType {
    Max,    ///< Selecciona el valor máximo en cada ventana de pooling.
    Average ///< Calcula el promedio de los valores en cada ventana de pooling.
  };

  /**
   * @brief Constructor de la capa Pooling2D.
   * @param poolSize El tamaño (altura y anchura) de la ventana de pooling.
   * @param type El tipo de pooling a realizar (Max o Average).
   * @param stride El paso con el que se desliza la ventana. Si es 0, se usa un stride igual a `poolSize`.
   */
  explicit Pooling2D(size_t poolSize, PoolType type = PoolType::Max, size_t stride = 0);

  /**
   * @brief Realiza el paso hacia adelante de la operación de pooling.
   * @param input Tensor de entrada de forma {Batch, Channels, Height, Width}.
   * @param isTraining Si es `true` y el tipo es Max, almacena los índices de los máximos.
   * @return Tensor de salida submuestreado de forma {Batch, Channels, outHeight, outWidth}.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza el paso hacia atrás de la operación de pooling.
   * @details Para Max Pooling, el gradiente se enruta solo a la neurona que fue el
   *          máximo. Para Average Pooling, el gradiente se distribuye
   *          uniformemente entre todas las neuronas de la ventana.
   * @param outputGradient Gradiente de la pérdida respecto a la salida de esta capa (dE/dY).
   * @return Gradiente de la pérdida respecto a la entrada de esta capa (dE/dX).
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Devuelve el nombre de la capa, incluyendo el tipo de pooling.
   * @return El string "MaxPooling" o "AveragePooling".
   * @override
   */
  std::string getName() const override;

private:
  PoolType type;   ///< El tipo de pooling (Max o Average).
  size_t poolSize; ///< Tamaño de la ventana de pooling.
  size_t stride;   ///< Paso de la ventana de pooling.

  // --- Estado para el backward pass ---

  /**
   * @brief Almacena los índices de los valores máximos para el backward pass de Max Pooling.
   *        Tiene la misma forma que la salida de la capa. Cada elemento contiene el
   *        índice plano (dentro de la ventana de pooling) del valor máximo.
   */
  Tensor maxIndices;

  /** @brief Almacena la forma del tensor de entrada para reconstruir el gradiente. */
  std::vector<size_t> inputShape;
};

#endif // POOLING2D_HPP
