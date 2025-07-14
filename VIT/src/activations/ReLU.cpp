#include "activations/ReLU.hpp"
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

ReLU::ReLU() {}

Tensor ReLU::forward(const Tensor &input, bool isTraining) {
  if (isTraining) {
    // Guarda la entrada para el calculo en backward.
    this->inputTensor = input;
  }

  Tensor result(input.getShape());
  const auto &shape = input.getShape();

  // Soporte para tensores 2D {batch, features}.
  if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        float val = input(i, j);
        result(i, j) = (val > 0) ? val : 0.0f;
      }
    }
  }
  // Soporte para tensores 3D {batch, tokens, features}.
  else if (shape.size() == 3) {
#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        for (size_t k = 0; k < shape[2]; ++k) {
          float val = input(i, j, k);
          result(i, j, k) = (val > 0) ? val : 0.0f;
        }
      }
    }
  } else {
    throw std::runtime_error("ReLU::forward solo soporta entradas 2D o 3D.");
  }

  return result;
}

Tensor ReLU::backward(const Tensor &outputGradient) {
  Tensor inputGradient(this->inputTensor.getShape());
  const auto &shape = this->inputTensor.getShape();

  // dE/dX = dE/dY * dY/dX. La derivada de ReLU (dY/dX) es 1 si X > 0, sino 0.
  if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        inputGradient(i, j) = (this->inputTensor(i, j) > 0) ? outputGradient(i, j) : 0.0f;
      }
    }
  } else if (shape.size() == 3) {
#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        for (size_t k = 0; k < shape[2]; ++k) {
          inputGradient(i, j, k) = (this->inputTensor(i, j, k) > 0) ? outputGradient(i, j, k) : 0.0f;
        }
      }
    }
  } else {
    throw std::runtime_error("ReLU::backward solo soporta entradas 2D o 3D.");
  }

  return inputGradient;
}
