#include "activations/GELU.hpp"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// Constante para la aproximacion de GELU: sqrt(2/pi).
const float SQRT_2_OVER_PI = 0.7978845608028654f;

GELU::GELU() {}

Tensor GELU::forward(const Tensor &input, bool isTraining) {
  if (isTraining) {
    // Guarda la entrada para el calculo en backward.
    this->inputTensor = input;
  }

  Tensor result(input.getShape());

  // Se asume que el tensor es contiguo para mayor rendimiento.
  if (input.isContiguous() && result.isContiguous()) {
    const float *in_data = input.getData();
    float *out_data = result.getData();
    size_t size = input.getSize();

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
      float x = in_data[i];
      // Aproximacion de GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      float x_cubed = x * x * x;
      float inner = SQRT_2_OVER_PI * (x + 0.044715f * x_cubed);
      out_data[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
  } else {
    throw std::runtime_error("GELU::forward solo implementado para tensores contiguos.");
  }

  return result;
}

Tensor GELU::backward(const Tensor &outputGradient) {
  Tensor inputGradient(inputTensor.getShape());

  // Se asume que los tensores son contiguos para mayor rendimiento.
  if (inputTensor.isContiguous() && outputGradient.isContiguous()) {
    const float *in_data = inputTensor.getData();
    const float *grad_out_data = outputGradient.getData();
    float *grad_in_data = inputGradient.getData();
    size_t size = inputTensor.getSize();

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
      float x = in_data[i];
      float x_squared = x * x;

      // Calculo de la derivada de la aproximacion de GELU.
      // dGELU/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d(inner)/dx
      // sech^2(z) = 1 - tanh^2(z)
      float inner = SQRT_2_OVER_PI * (x + 0.044715f * x_squared * x);
      float tanh_inner = std::tanh(inner);

      float d_inner_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * 0.044715f * x_squared);
      float sech_squared = 1.0f - tanh_inner * tanh_inner;

      float dGELU_dx = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_squared * d_inner_dx;

      // Aplicacion de la regla de la cadena: dE/dX = dE/dY * dY/dX
      grad_in_data[i] = dGELU_dx * grad_out_data[i];
    }
  } else {
    throw std::runtime_error("GELU::backward solo implementado para tensores contiguos.");
  }

  return inputGradient;
}
