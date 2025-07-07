#include "activations/ReLU.hpp"

#include <stdexcept>

// Incluir OpenMP si está disponible
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Constructor de la capa ReLU. No requiere inicialización.
 */
ReLU::ReLU() {}

/**
 * @brief Aplica la función de activación ReLU: f(x) = max(0, x).
 */
Tensor ReLU::forward(const Tensor &input, bool isTraining) {
  // Si estamos entrenando, guardamos la entrada original. La necesitaremos
  // en el backward pass para saber por dónde puede fluir el gradiente.
  if (isTraining) {
    this->inputTensor = input;
  }

  Tensor result(input.getShape());
  const auto &shape = input.getShape();

  // Especialización para las formas más comunes (2D para Dense, 4D para Conv)
  if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        float val = input(i, j);
        result(i, j) = (val > 0) ? val : 0.0f;
      }
    }
  } else if (shape.size() == 4) {
#pragma omp parallel for collapse(4)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t c = 0; c < shape[1]; ++c) {
        for (size_t h = 0; h < shape[2]; ++h) {
          for (size_t w = 0; w < shape[3]; ++w) {
            float val = input(b, c, h, w);
            result(b, c, h, w) = (val > 0) ? val : 0.0f;
          }
        }
      }
    }
  } else {
    throw std::runtime_error("ReLU::forward solo soporta entradas 2D o 4D.");
  }

  return result;
}

/**
 * @brief Calcula el gradiente para la capa ReLU.
 */
Tensor ReLU::backward(const Tensor &outputGradient) {
  // La derivada de ReLU es una función escalón:
  // - d(ReLU)/dx = 1 si x > 0
  // - d(ReLU)/dx = 0 si x <= 0
  // Por la regla de la cadena, el gradiente de entrada es el gradiente de
  // salida multiplicado por esta derivada.
  Tensor inputGradient(this->inputTensor.getShape());
  const auto &shape = this->inputTensor.getShape();

  if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        // Si la entrada original era > 0, el gradiente pasa. Si no, se bloquea (es 0).
        inputGradient(i, j) = (this->inputTensor(i, j) > 0) ? outputGradient(i, j) : 0.0f;
      }
    }
  } else if (shape.size() == 4) {
#pragma omp parallel for collapse(4)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t c = 0; c < shape[1]; ++c) {
        for (size_t h = 0; h < shape[2]; ++h) {
          for (size_t w = 0; w < shape[3]; ++w) {
            inputGradient(b, c, h, w) = (this->inputTensor(b, c, h, w) > 0) ? outputGradient(b, c, h, w) : 0.0f;
          }
        }
      }
    }
  } else {
    throw std::runtime_error("ReLU::backward solo soporta entradas 2D o 4D.");
  }

  return inputGradient;
}
