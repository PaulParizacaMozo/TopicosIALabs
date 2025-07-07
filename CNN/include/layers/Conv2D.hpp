#ifndef CONV2D_HPP
#define CONV2D_HPP

#include "layers/Layer.hpp"

/**
 * @class Conv2D
 * @brief Una capa de convolución 2D, el bloque de construcción fundamental de las CNN.
 *
 * Esta capa aplica un conjunto de filtros (kernels) aprendibles a una entrada
 * para producir mapas de características (feature maps). La convolución es una
 * operación clave para detectar patrones locales como bordes, texturas o formas.
 *
 * La implementación utiliza la técnica 'im2col' (image-to-column) para transformar
 * la operación de convolución en una multiplicación de matrices de alta eficiencia,
 * lo que permite aprovechar las optimizaciones de GEMM (General Matrix Multiply).
 */
class Conv2D : public Layer {
public:
  /**
   * @brief Constructor de la capa Conv2D.
   * @param inChannels Número de canales en el tensor de entrada (ej. 1 para escala de grises, 3 para RGB).
   * @param outChannels Número de filtros a aplicar, que determina la profundidad (canales) del tensor de salida.
   * @param kernelSize Tamaño (altura y anchura) de los filtros cuadrados (ej. 3 para un filtro 3x3).
   * @param stride El paso con el que el filtro se desliza sobre la entrada.
   * @param padding El número de píxeles de relleno (con ceros) que se añaden a los bordes de la entrada.
   */
  Conv2D(size_t inChannels, size_t outChannels, size_t kernelSize, size_t stride = 1, size_t padding = 0);

  /**
   * @brief Realiza el paso hacia adelante de la convolución.
   * @details Utiliza `im2col` para remodelar la entrada, luego realiza una
   *          multiplicación de matrices con los pesos y finalmente añade el bias.
   * @param input Tensor de entrada de forma {Batch, inChannels, Height, Width}.
   * @param isTraining Si es `true`, almacena datos necesarios para el backward pass.
   * @return Tensor de salida (mapas de características) de forma {Batch, outChannels, outHeight, outWidth}.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza el paso hacia atrás de la convolución.
   * @details Calcula los gradientes para los pesos, el bias y el tensor de entrada
   *          utilizando la transpuesta de la convolución (implementada con `col2im`).
   * @param outputGradient Gradiente de la pérdida respecto a la salida de esta capa (dE/dY).
   * @return Gradiente de la pérdida respecto a la entrada de esta capa (dE/dX).
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Devuelve punteros a los parámetros entrenables (pesos y bias).
   * @return Un vector con punteros a `weights` y `bias`.
   * @override
   */
  std::vector<Tensor *> getParameters() override;

  /**
   * @brief Devuelve punteros a los gradientes de los parámetros.
   * @return Un vector con punteros a `weightGradients` y `biasGradients`.
   * @override
   */
  std::vector<Tensor *> getGradients() override;

  std::string getName() const override { return "Conv2D"; }

private:
  // --- Hiperparámetros de la capa ---
  size_t inChannels;
  size_t outChannels;
  size_t kernelSize;
  size_t stride;
  size_t padding;

  // --- Parámetros entrenables ---
  Tensor weights; ///< Pesos de los filtros. Forma: {outChannels, inChannels, kernelSize, kernelSize}.
  Tensor bias;    ///< Biases de los filtros. Forma: {1, outChannels}. Un bias por filtro.

  // --- Gradientes ---
  Tensor weightGradients; ///< Gradientes de los pesos.
  Tensor biasGradients;   ///< Gradientes de los biases.

  // --- Estado para el backward pass ---
  Tensor im2colMatrix;            ///< Matriz generada por `im2col` en el forward pass. Se reutiliza en el backward.
  std::vector<size_t> inputShape; ///< Forma del tensor de entrada, necesaria para `col2im`.

  // --- Funciones de utilidad para la convolución ---

  /**
   * @brief Transforma los parches de la imagen de entrada en columnas de una matriz.
   * @details Es la clave para convertir la convolución en una multiplicación de matrices.
   * @param input Tensor de entrada (con padding si es necesario).
   * @param outH Altura de la salida calculada.
   * @param outW Anchura de la salida calculada.
   */
  void im2col(const Tensor &input, size_t outH, size_t outW);

  /**
   * @brief Transforma las columnas de una matriz de gradientes de vuelta a una "imagen" de gradientes.
   * @details Es la operación inversa de `im2col` y se usa para calcular dE/dX.
   * @param colMatrix Matriz de columnas (gradientes) que se va a transformar.
   * @param outputImage Tensor de destino donde se acumulan los gradientes de la "imagen".
   */
  void col2im(const Tensor &colMatrix, Tensor &outputImage);
};

#endif // CONV2D_HPP
