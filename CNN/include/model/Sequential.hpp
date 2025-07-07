#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "layers/Layer.hpp"
#include "losses/Loss.hpp"
#include "optimizers/Optimizer.hpp"

#include <memory>  // Para std::unique_ptr
#include <utility> // Para std::pair, std::forward
#include <vector>

/**
 * @class Sequential
 * @brief Representa un modelo de red neuronal como una pila lineal de capas.
 *
 * Esta clase orquesta todo el proceso de la red neuronal: construcción del
 * modelo, entrenamiento, evaluación y predicción. Gestiona la vida de las
 * capas, la función de pérdida y el optimizador, y define el bucle de
 * entrenamiento principal.
 */
class Sequential {
public:
  /** @brief Constructor por defecto. */
  Sequential();

  /** @brief Destructor por defecto.
   *  @details Gracias a `std::unique_ptr`, liberará automáticamente la memoria
   *           de todas las capas, el optimizador y la función de pérdida.
   */
  ~Sequential() = default;

  /**
   * @brief Añade una capa al modelo usando plantillas variádicas.
   * @details Permite construir y añadir una capa en una sola línea, pasando
   *          directamente los argumentos de su constructor.
   * @tparam LayerType El tipo de la capa a añadir (ej. Dense, Conv2D).
   * @tparam Args Los tipos de los argumentos del constructor de la capa.
   * @param args Los argumentos para construir la capa.
   * @code
   * model.add<Dense>(784, 128);
   * model.add<ReLU>();
   * @endcode
   */
  template <typename LayerType, typename... Args> void add(Args &&...args);

  /**
   * @brief Configura el modelo para el entrenamiento.
   * @details Especifica el optimizador y la función de pérdida que se usarán.
   * @tparam OptimizerType El tipo del optimizador (ej. SGD, Adam).
   * @tparam LossType El tipo de la función de pérdida (ej. CrossEntropy).
   * @tparam Args Los tipos de los argumentos del constructor del optimizador.
   * @param optimizerArgs Los argumentos para construir el optimizador (ej. learning_rate).
   */
  template <typename OptimizerType, typename LossType, typename... Args> void compile(Args &&...optimizerArgs);

  /**
   * @brief Entrena el modelo usando los datos proporcionados.
   * @details Itera sobre los datos durante un número de épocas, dividiéndolos
   *          en mini-batches, y realiza los pasos de forward, backward y
   *          actualización de pesos en cada uno. Muestra el progreso en la
   *          consola.
   * @param X_train Tensor con los datos de entrenamiento.
   * @param y_train Tensor con las etiquetas de entrenamiento.
   * @param epochs El número total de veces que se iterará sobre todo el dataset.
   * @param batchSize El número de muestras por actualización de gradiente.
   * @param X_val Tensor con los datos de validación.
   * @param y_val Tensor con las etiquetas de validación.
   */
  void train(const Tensor &X_train, const Tensor &y_train, int epochs, size_t batchSize, const Tensor &X_val,
             const Tensor &y_val);

  /**
   * @brief Evalúa el rendimiento del modelo en un conjunto de datos.
   * @details Calcula la pérdida y la precisión del modelo en los datos proporcionados
   *          sin realizar retropropagación ni actualizar los pesos.
   * @param X Tensor con los datos de evaluación.
   * @param y Tensor con las etiquetas de evaluación.
   * @return Un par `{pérdida, precisión}`. La precisión es un valor entre 0 y 1.
   */
  std::pair<float, float> evaluate(const Tensor &X, const Tensor &y);

  /**
   * @brief Genera predicciones para un conjunto de datos de entrada.
   * @details Realiza un único paso hacia adelante a través de toda la red.
   * @param input El tensor de entrada para el que se generarán predicciones.
   * @return Un tensor con las predicciones del modelo (logits).
   */
  Tensor predict(const Tensor &input);

  /**
   * @brief Recopila los punteros a los parámetros de todas las capas.
   * @details Usado internamente para pasar los parámetros al optimizador.
   * @return Un vector de punteros a todos los tensores de parámetros entrenables.
   */
  std::vector<Tensor *> getParameters();

private:
  /// La pila de capas que componen el modelo.
  std::vector<std::unique_ptr<Layer>> layers;

  /// El optimizador que actualizará los pesos del modelo.
  std::unique_ptr<Optimizer> optimizer;

  /// La función de pérdida que medirá el error del modelo.
  std::unique_ptr<Loss> loss;
};

// --- Implementación de las plantillas en el header ---

template <typename LayerType, typename... Args> void Sequential::add(Args &&...args) {
  // Crea un puntero único a la capa, pasando los argumentos al constructor
  // de LayerType. `std::forward` preserva la categoría del valor (lvalue/rvalue).
  auto layer = std::make_unique<LayerType>(std::forward<Args>(args)...);
  this->layers.push_back(std::move(layer));
}

template <typename OptimizerType, typename LossType, typename... Args> void Sequential::compile(Args &&...optimizerArgs) {
  // Crea el optimizador, pasando sus argumentos (ej. learning rate).
  this->optimizer = std::make_unique<OptimizerType>(std::forward<Args>(optimizerArgs)...);

  // Crea la función de pérdida. Se asume que no tiene argumentos por ahora.
  this->loss = std::make_unique<LossType>();
}

#endif // SEQUENTIAL_HPP
