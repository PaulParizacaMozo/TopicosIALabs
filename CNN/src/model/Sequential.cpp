#include "model/Sequential.hpp"
#include "losses/CrossEntropy.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// --- Implementación de los Métodos ---

/**
 * @brief Constructor por defecto del modelo secuencial.
 */
Sequential::Sequential() {}

/**
 * @brief Realiza una pasada hacia adelante a través de toda la red para hacer una predicción.
 * @details No se calcula ni se almacena ninguna información para la retropropagación.
 *          `isTraining` se pasa como `false` a todas las capas.
 */
Tensor Sequential::predict(const Tensor &input) {
  Tensor currentOutput = input;
  // Propaga la salida de una capa como la entrada de la siguiente.
  for (const auto &layer : this->layers) {
    currentOutput = layer->forward(currentOutput, false); // isTraining = false
  }
  return currentOutput;
}

/**
 * @brief Función auxiliar para encontrar el índice del valor máximo en un array.
 * @details Utilizado para determinar la clase predicha a partir de un vector de probabilidades.
 */
size_t argmax(const float *data, size_t size) {
  // Encuentra un iterador al elemento máximo y calcula su distancia desde el inicio.
  return std::distance(data, std::max_element(data, data + size));
}

/**
 * @brief Evalúa la pérdida y la precisión del modelo en un conjunto de datos.
 * @details Procesa los datos en batches para no agotar la memoria.
 */
std::pair<float, float> Sequential::evaluate(const Tensor &X, const Tensor &y) {
  if (!loss) {
    throw std::runtime_error("El modelo debe ser compilado para poder evaluar.");
  }

  const size_t numSamples = X.getShape()[0];
  const size_t numClasses = y.getShape()[1];
  float totalLoss = 0.0f;
  size_t correctPredictions = 0;

  // Usar un tamaño de batch fijo para la evaluación para consistencia.
  const size_t evalBatchSize = 256;
  size_t numBatches = 0;

  for (size_t i = 0; i < numSamples; i += evalBatchSize) {
    size_t end = std::min(i + evalBatchSize, numSamples);

    Tensor X_batch = X.slice(i, end - i);
    Tensor y_batch = y.slice(i, end - i);

    // 1. Obtener predicciones (logits) del modelo.
    Tensor yPred = this->predict(X_batch);

    // 2. Calcular la pérdida del batch y acumularla.
    totalLoss += this->loss->calculate(yPred, y_batch);

    // 3. Calcular la precisión del batch.
    Tensor probabilities = softmax(yPred);
    for (size_t sample_idx = 0; sample_idx < X_batch.getShape()[0]; ++sample_idx) {
      // Punteros directos para evitar recálculos de acceso en `operator()`
      const float *predProbsPtr = probabilities.getData() + sample_idx * numClasses;
      const float *trueLabelsPtr = y.getData() + (i + sample_idx) * numClasses;

      size_t predictedClass = argmax(predProbsPtr, numClasses);
      size_t trueClass = argmax(trueLabelsPtr, numClasses);

      if (predictedClass == trueClass) {
        correctPredictions++;
      }
    }
    numBatches++;
  }

  // Calcular métricas promedio.
  float avgLoss = totalLoss / numBatches;
  float accuracy = static_cast<float>(correctPredictions) / numSamples;

  return {avgLoss, accuracy};
}

/**
 * @brief El bucle de entrenamiento principal del modelo.
 */
void Sequential::train(const Tensor &X_train, const Tensor &y_train, int epochs, size_t batchSize, const Tensor &X_val,
                       const Tensor &y_val) {
  if (!optimizer || !loss) {
    throw std::runtime_error("El modelo debe ser compilado antes de entrenar.");
  }

  const size_t numTrainSamples = X_train.getShape()[0];
  const size_t numClasses = y_train.getShape()[1]; // Asume clasificación

  for (int epoch = 0; epoch < epochs; ++epoch) {
    auto epochStart = std::chrono::high_resolution_clock::now();
    float epochTrainLoss = 0.0f;
    size_t epochTrainCorrect = 0;
    size_t numBatches = 0;

    // --- Bucle principal sobre los mini-batches ---
    for (size_t i = 0; i < numTrainSamples; i += batchSize) {
      size_t end = std::min(i + batchSize, numTrainSamples);
      Tensor X_batch = X_train.slice(i, end - i);
      Tensor y_batch = y_train.slice(i, end - i);

      // --- 1. Forward Pass ---
      // Propaga la entrada a través de la red, capa por capa, con `isTraining=true`.
      // Esto asegura que las capas (como Dropout, ReLU) almacenen lo necesario.
      Tensor yPred = X_batch;
      for (const auto &layer : this->layers) {
        yPred = layer->forward(yPred, true);
      }

      // --- 2. Cálculo de Pérdida y Métricas ---
      // Se usan los logits (yPred) para calcular la pérdida y la precisión.
      epochTrainLoss += this->loss->calculate(yPred, y_batch);
      Tensor probabilities = softmax(yPred);
      for (size_t sample_idx = 0; sample_idx < X_batch.getShape()[0]; ++sample_idx) {
        const float *predProbsPtr = probabilities.getData() + sample_idx * numClasses;
        const float *trueLabelsPtr = y_train.getData() + (i + sample_idx) * numClasses;
        if (argmax(predProbsPtr, numClasses) == argmax(trueLabelsPtr, numClasses)) {
          epochTrainCorrect++;
        }
      }

      // --- 3. Backward Pass (Retropropagación) ---
      // Inicia la retropropagación desde la función de pérdida.
      Tensor gradient = this->loss->backward(yPred, y_batch);
      // Propaga el gradiente hacia atrás a través de la red, en orden inverso.
      for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
        gradient = (*it)->backward(gradient);
      }

      // --- 4. Actualización de Pesos ---
      // Recopila todos los parámetros y gradientes de todas las capas.
      std::vector<Tensor *> allParams;
      std::vector<Tensor *> allGrads;
      for (const auto &layer : this->layers) {
        auto params = layer->getParameters();
        auto grads = layer->getGradients();
        allParams.insert(allParams.end(), params.begin(), params.end());
        allGrads.insert(allGrads.end(), grads.begin(), grads.end());
      }
      // Pasa los parámetros y gradientes al optimizador para que los actualice.
      if (!allParams.empty()) {
        this->optimizer->update(allParams, allGrads);
      }

      numBatches++;
    }

    // --- Fin de la Época: Evaluación y Reporte ---
    auto [valLoss, valAcc] = this->evaluate(X_val, y_val);
    auto epochEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> epochDuration = epochEnd - epochStart;

    float trainLoss = epochTrainLoss / numBatches;
    float trainAcc = static_cast<float>(epochTrainCorrect) / numTrainSamples;

    // --- Impresión del Resumen de la Época ---
    std::cout << "Epoca " << std::setw(2) << epoch + 1 << "/" << epochs << " - Tiempo: " << std::fixed << std::setprecision(2)
              << epochDuration.count() << "s"
              << " | train_loss: " << std::fixed << std::setprecision(4) << trainLoss << " | train_acc: " << std::fixed
              << std::setprecision(4) << trainAcc << " | val_loss: " << std::fixed << std::setprecision(4) << valLoss
              << " | val_acc: " << std::fixed << std::setprecision(4) << valAcc << std::endl;
  }
}

/**
 * @brief Recopila los punteros a todos los parámetros entrenables del modelo.
 */
std::vector<Tensor *> Sequential::getParameters() {
  std::vector<Tensor *> allParams;
  for (const auto &layer : this->layers) {
    auto params = layer->getParameters();
    // Añade los parámetros de esta capa a la lista global.
    allParams.insert(allParams.end(), params.begin(), params.end());
  }
  return allParams;
}
