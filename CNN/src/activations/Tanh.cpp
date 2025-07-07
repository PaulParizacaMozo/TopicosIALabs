#include "activations/Tanh.hpp"
#include <cmath>
#include <omp.h>
#include <stdexcept>

Tanh::Tanh() {}

Tensor Tanh::forward(const Tensor &input, bool isTraining) {
  Tensor result(input.getShape());
  const auto &shape = input.getShape();

  if (shape.size() == 4) {
#pragma omp parallel for collapse(4)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t c = 0; c < shape[1]; ++c) {
        for (size_t h = 0; h < shape[2]; ++h) {
          for (size_t w = 0; w < shape[3]; ++w) {
            result(b, c, h, w) = std::tanh(input(b, c, h, w));
          }
        }
      }
    }
  } else if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        result(i, j) = std::tanh(input(i, j));
      }
    }
  } else {
    throw std::runtime_error("Tanh::forward solo soporta entradas 2D o 4D.");
  }

  if (isTraining) {
    this->outputTensor = result;
  }

  return result;
}

Tensor Tanh::backward(const Tensor &outputGradient) {
  Tensor inputGradient(this->outputTensor.getShape());
  const auto &shape = this->outputTensor.getShape();

  if (shape.size() == 4) {
#pragma omp parallel for collapse(4)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t c = 0; c < shape[1]; ++c) {
        for (size_t h = 0; h < shape[2]; ++h) {
          for (size_t w = 0; w < shape[3]; ++w) {
            float tanh_val = this->outputTensor(b, c, h, w);
            inputGradient(b, c, h, w) = outputGradient(b, c, h, w) * (1.0f - (tanh_val * tanh_val));
          }
        }
      }
    }
  } else if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        float tanh_val = this->outputTensor(i, j);
        inputGradient(i, j) = outputGradient(i, j) * (1.0f - (tanh_val * tanh_val));
      }
    }
  } else {
    throw std::runtime_error("Tanh::backward solo soporta entradas 2D o 4D.");
  }

  return inputGradient;
}
