#include "activations/Sigmoid.hpp"

#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Constructor de la capa Sigmoid. No requiere inicialización.
 */
Sigmoid::Sigmoid() {}

/**
 * @brief Aplica la función de activación sigmoide: f(x) = 1 / (1 + exp(-x)).
 */
Tensor Sigmoid::forward(const Tensor &input, bool isTraining) {
  Tensor result(input.getShape());
  const auto &shape = input.getShape();

  // Especialización para formas 2D y 4D.
  if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        result(i, j) = 1.0f / (1.0f + std::exp(-input(i, j)));
      }
    }
  } else if (shape.size() == 4) {
#pragma omp parallel for collapse(4)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t c = 0; c < shape[1]; ++c) {
        for (size_t h = 0; h < shape[2]; ++h) {
          for (size_t w = 0; w < shape[3]; ++w) {
            result(b, c, h, w) = 1.0f / (1.0f + std::exp(-input(b, c, h, w)));
          }
        }
      }
    }
  } else {
    throw std::runtime_error("Sigmoid::forward solo soporta entradas 2D o 4D.");
  }

  // Si estamos entrenando, guardamos la *salida* calculada.
  // Esto es una optimización, ya que la derivada se puede calcular desde la salida.
  if (isTraining) {
    this->outputTensor = result;
  }

  return result;
}

/**
 * @brief Calcula el gradiente para la capa Sigmoid.
 */
Tensor Sigmoid::backward(const Tensor &outputGradient) {
  // La derivada del sigmoide es: f'(x) = f(x) * (1 - f(x))
  // donde f(x) es el valor de la salida del sigmoide.
  // Por la regla de la cadena: dE/dX = dE/dY * f'(x)
  Tensor inputGradient(this->outputTensor.getShape());
  const auto &shape = this->outputTensor.getShape();

  if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        float sigmoid_val = this->outputTensor(i, j);
        float local_gradient = sigmoid_val * (1.0f - sigmoid_val);
        inputGradient(i, j) = outputGradient(i, j) * local_gradient;
      }
    }
  } else if (shape.size() == 4) {
#pragma omp parallel for collapse(4)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t c = 0; c < shape[1]; ++c) {
        for (size_t h = 0; h < shape[2]; ++h) {
          for (size_t w = 0; w < shape[3]; ++w) {
            float sigmoid_val = this->outputTensor(b, c, h, w);
            float local_gradient = sigmoid_val * (1.0f - sigmoid_val);
            inputGradient(b, c, h, w) = outputGradient(b, c, h, w) * local_gradient;
          }
        }
      }
    }
  } else {
    throw std::runtime_error("Sigmoid::backward solo soporta entradas 2D o 4D.");
  }

  return inputGradient;
}
