#include "losses/CrossEntropy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

Tensor softmax(const Tensor &logits) {
  Tensor probabilities(logits.getShape());
  const size_t batchSize = logits.getShape()[0];
  const size_t numClasses = logits.getShape()[1];

#pragma omp parallel for
  for (size_t i = 0; i < batchSize; ++i) {
    // 1. Encontrar el logit maximo para estabilidad numerica.
    float maxLogit = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < numClasses; ++j) {
      if (logits(i, j) > maxLogit) {
        maxLogit = logits(i, j);
      }
    }

    // 2. Calcular los exponenciales y su suma.
    float sumExp = 0.0f;
    for (size_t j = 0; j < numClasses; ++j) {
      float expVal = std::exp(logits(i, j) - maxLogit);
      probabilities(i, j) = expVal;
      sumExp += expVal;
    }

    // 3. Normalizar para obtener las probabilidades.
    for (size_t j = 0; j < numClasses; ++j) {
      probabilities(i, j) /= sumExp;
    }
  }
  return probabilities;
}

// Calcula la perdida de entropia cruzada para un batch.
float CrossEntropy::calculate(const Tensor &yPred, const Tensor &yTrue) {
  if (yPred.getShape() != yTrue.getShape()) {
    throw std::runtime_error("Las formas de prediccion y etiquetas no coinciden.");
  }

  // 1. Convertir los logits en probabilidades y guardar para backward.
  this->softmaxOutput = softmax(yPred);

  // 2. Calcular la perdida.
  const size_t batchSize = yPred.getShape()[0];
  const size_t numClasses = yPred.getShape()[1];
  float totalLoss = 0.0f;
  const float epsilon = 1e-12; // Para evitar log(0).

#pragma omp parallel for reduction(+ : totalLoss)
  for (size_t i = 0; i < batchSize; ++i) {
    for (size_t j = 0; j < numClasses; ++j) {
      // La perdida solo se calcula para la clase correcta (donde yTrue es 1).
      if (yTrue(i, j) == 1.0f) {
        totalLoss += -std::log(this->softmaxOutput(i, j) + epsilon);
      }
    }
  }

  // Devuelve la perdida promedio por muestra.
  return totalLoss / batchSize;
}

// Calcula el gradiente de (Softmax + CrossEntropy) respecto a los logits.
Tensor CrossEntropy::backward(const Tensor & /*yPred*/, const Tensor &yTrue) {
  // El gradiente es: (softmax(yPred) - yTrue) / batchSize
  // Reutilizamos softmaxOutput calculado en `calculate`.
  Tensor gradient = this->softmaxOutput;

  const size_t batchSize = yTrue.getShape()[0];
  const size_t numClasses = yTrue.getShape()[1];

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < batchSize; ++i) {
    for (size_t j = 0; j < numClasses; ++j) {
      // Se normaliza por el tamaño del batch para que la magnitud del gradiente
      // sea independiente del tamaño del lote.
      gradient(i, j) = (gradient(i, j) - yTrue(i, j)) / batchSize;
    }
  }

  return gradient;
}
