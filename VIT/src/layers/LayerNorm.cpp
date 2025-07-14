#include "layers/LayerNorm.hpp"
#include <cmath>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

LayerNorm::LayerNorm(size_t featureSize, float epsilon) : featureSize(featureSize), epsilon(epsilon) {
  // Gamma se inicializa a 1 (sin cambio de escala inicial).
  this->gamma = Tensor({1, featureSize});
  this->gamma.fill(1.0f);

  // Beta se inicializa a 0 (sin desplazamiento inicial).
  this->beta = Tensor({1, featureSize});
  this->beta.fill(0.0f);

  // Gradientes se inicializan a cero.
  this->gammaGradient = Tensor({1, featureSize});
  this->betaGradient = Tensor({1, featureSize});
}

Tensor LayerNorm::forward(const Tensor &input, bool isTraining) {
  const auto &inputShape = input.getShape();
  if (inputShape.back() != this->featureSize) {
    throw std::runtime_error("La ultima dimension de entrada no coincide con featureSize.");
  }

  // El "batch" es el producto de todas las dimensiones excepto la ultima.
  size_t batchSize = input.getSize() / this->featureSize;

  // Aplanamos temporalmente la entrada a 2D para simplificar calculos.
  Tensor input2D = input.reshape({batchSize, this->featureSize});

  // En entrenamiento, guardamos valores intermedios para backward.
  if (isTraining) {
    this->inputTensor = input2D;
    this->mean = Tensor({batchSize, 1});
    this->variance = Tensor({batchSize, 1}); // Se reutilizara para guardar inv_stddev.
  }

  Tensor output2D({batchSize, this->featureSize});
  this->normalizedInput = Tensor({batchSize, this->featureSize});

#pragma omp parallel for
  for (size_t i = 0; i < batchSize; ++i) {
    // --- 1. Calcular la media ---
    float current_mean = 0.0f;
    for (size_t j = 0; j < this->featureSize; ++j) {
      current_mean += input2D(i, j);
    }
    current_mean /= this->featureSize;

    // --- 2. Calcular la varianza ---
    float current_variance = 0.0f;
    for (size_t j = 0; j < this->featureSize; ++j) {
      float diff = input2D(i, j) - current_mean;
      current_variance += diff * diff;
    }
    current_variance /= this->featureSize;

    float inv_stddev = 1.0f / std::sqrt(current_variance + this->epsilon);

    if (isTraining) {
      this->mean(i, 0) = current_mean;
      this->variance(i, 0) = inv_stddev; // Guardamos 1/sqrt(var+eps)
    }

    // --- 3. Normalizar, escalar y desplazar ---
    for (size_t j = 0; j < this->featureSize; ++j) {
      float x_hat = (input2D(i, j) - current_mean) * inv_stddev;
      if (isTraining)
        this->normalizedInput(i, j) = x_hat; // Guardamos la entrada normalizada.

      output2D(i, j) = this->gamma(0, j) * x_hat + this->beta(0, j);
    }
  }

  // Devolvemos el tensor a su forma original.
  return output2D.reshape(inputShape);
}

Tensor LayerNorm::backward(const Tensor &outputGradient) {
  const auto &gradShape = outputGradient.getShape();
  size_t batchSize = outputGradient.getSize() / this->featureSize;

  // Aplanamos el gradiente de salida a 2D.
  Tensor grad2D = outputGradient.reshape({batchSize, this->featureSize});

  // Reseteamos los gradientes de los parametros antes de acumular.
  this->gammaGradient.fill(0.0f);
  this->betaGradient.fill(0.0f);
  Tensor inputGradient({batchSize, this->featureSize});

  // El bucle sobre el batch es secuencial para evitar race conditions al acumular
  // los gradientes de gamma y beta, que son compartidos por todo el batch.
  for (size_t i = 0; i < batchSize; ++i) {
    float inv_stddev = this->variance(i, 0); // Reutilizamos el valor guardado.

    float dL_dXhat_sum = 0;
    float dL_dXhat_dot_Xhat_sum = 0;

    // --- 1. Calcular gradientes de gamma, beta y sumas intermedias ---
    // dL/dgamma = sum(dL/dY * X_hat) ; dL/dbeta = sum(dL/dY)
    for (size_t j = 0; j < this->featureSize; ++j) {
      float grad_y_ij = grad2D(i, j);
      float x_hat_ij = this->normalizedInput(i, j);

      this->gammaGradient(0, j) += grad_y_ij * x_hat_ij;
      this->betaGradient(0, j) += grad_y_ij;

      float dL_dXhat = grad_y_ij * this->gamma(0, j);
      dL_dXhat_sum += dL_dXhat;
      dL_dXhat_dot_Xhat_sum += dL_dXhat * x_hat_ij;
    }

    // --- 2. Calcular el gradiente de la entrada (dL/dX) ---
    // Se aplica la formula completa derivada de la normalizacion.
    for (size_t j = 0; j < this->featureSize; ++j) {
      float dL_dXhat_ij = grad2D(i, j) * this->gamma(0, j);
      float x_hat_ij = this->normalizedInput(i, j);

      float term1 = this->featureSize * dL_dXhat_ij;
      float term2 = dL_dXhat_sum;
      float term3 = x_hat_ij * dL_dXhat_dot_Xhat_sum;

      inputGradient(i, j) = (1.0f / this->featureSize) * inv_stddev * (term1 - term2 - term3);
    }
  }

  // Devolvemos el gradiente a su forma original.
  return inputGradient.reshape(gradShape);
}

std::vector<Tensor *> LayerNorm::getParameters() { return {&this->gamma, &this->beta}; }

std::vector<Tensor *> LayerNorm::getGradients() { return {&this->gammaGradient, &this->betaGradient}; }
