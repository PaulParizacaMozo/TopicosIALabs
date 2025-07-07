#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "layers/Layer.hpp"
#include <cmath>

/**
 * @class Sigmoid
 * @brief Implementa la función de activación sigmoide (o logística).
 *
 * La función sigmoide mapea cualquier valor de entrada a un valor en el
 * rango (0, 1). Su fórmula es:
 *   `f(x) = 1 / (1 + exp(-x))`
 *
 * Históricamente fue muy popular, aunque hoy en día se usa con menos frecuencia
 * en las capas ocultas (donde ReLU es preferida), pero sigue siendo útil en
 * la capa de salida para problemas de clasificación binaria.
 */
class Sigmoid : public Layer {
public:
  /** @brief Constructor por defecto. */
  Sigmoid();

  /**
   * @brief Aplica la función sigmoide elemento a elemento.
   * @param input El tensor de entrada.
   * @param isTraining Si es `true`, guarda la salida para el backward pass.
   * @return Un nuevo tensor con la función sigmoide aplicada.
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza el paso hacia atrás para la función sigmoide.
   * @details Utiliza la propiedad de que la derivada del sigmoide, f'(x), se
   *          puede expresar como f(x) * (1 - f(x)), donde f(x) es la salida
   *          del forward pass.
   * @param outputGradient El gradiente de la pérdida respecto a la salida.
   * @return El gradiente de la pérdida respecto a la entrada.
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Devuelve el nombre de la capa.
   * @return El string "Sigmoid".
   * @override
   */
  std::string getName() const override { return "Sigmoid"; }

private:
  /**
   * @brief Almacena la salida del forward pass.
   *        Es más eficiente que guardar la entrada, ya que la derivada del
   *        sigmoide se calcula directamente a partir de su salida.
   */
  Tensor outputTensor;
};

#endif // SIGMOID_HPP
