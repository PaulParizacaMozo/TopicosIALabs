#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include "layers/Layer.hpp"

/**
 * @class Dropout
 * @brief Una capa de regularización que previene el sobreajuste (overfitting).
 *
 * Durante el entrenamiento, la capa Dropout pone a cero aleatoriamente una
 * fracción (`rate`) de las unidades de entrada y escala las restantes para
 * mantener la suma de las activaciones constante en expectativa. Esto fuerza a
 * la red a aprender características más robustas y a no depender de neuronas
 * individuales.
 *
 * Durante la inferencia (`isTraining = false`), esta capa no realiza ninguna
 * operación y simplemente pasa la entrada a la salida sin modificarla.
 * Esta implementación utiliza la técnica de "inverted dropout".
 */
class Dropout : public Layer {
public:
  /**
   * @brief Constructor de la capa Dropout.
   * @param rate La fracción de unidades de entrada que se pondrán a cero.
   *             Debe ser un valor en el rango [0, 1).
   */
  explicit Dropout(float rate);

  /**
   * @brief Aplica la máscara de dropout a la entrada durante el entrenamiento.
   * @param input El tensor de entrada.
   * @param isTraining Si es `true`, se aplica el dropout. Si es `false`, la
   *        capa no hace nada.
   * @return El tensor de salida con dropout aplicado (si aplica).
   * @override
   */
  Tensor forward(const Tensor &input, bool isTraining) override;

  /**
   * @brief Realiza el paso hacia atrás, aplicando la misma máscara de dropout.
   * @details El gradiente solo fluye a través de las neuronas que no fueron
   *          desactivadas durante el forward pass.
   * @param outputGradient El gradiente de la pérdida respecto a la salida.
   * @return El gradiente de la pérdida respecto a la entrada.
   * @override
   */
  Tensor backward(const Tensor &outputGradient) override;

  /**
   * @brief Devuelve el nombre de la capa.
   * @return El string "Dropout".
   * @override
   */
  std::string getName() const override { return "Dropout"; }

private:
  float rate;         ///< La probabilidad de que una unidad sea puesta a cero.
  float scale;        ///< Factor de escala para las unidades restantes (1.0 / (1.0 - rate)).
  Tensor dropoutMask; ///< Máscara binaria (con escalado) que se aplica a la entrada.
};

#endif // DROPOUT_HPP
