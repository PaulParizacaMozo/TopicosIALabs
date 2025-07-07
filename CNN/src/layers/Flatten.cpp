#include "layers/Flatten.hpp"

// Incluir OpenMP si está disponible
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Constructor de la capa Flatten. No requiere inicialización.
 */
Flatten::Flatten() {
  // No se necesita ninguna acción en el constructor.
  // El estado (inputShape) se establece en el forward pass.
}

/**
 * @brief Realiza el paso hacia adelante, aplanando la entrada.
 */
Tensor Flatten::forward(const Tensor &input, bool /* isTraining */) {
  // 1. Almacenar la forma de la entrada para el backward pass.
  this->inputShape = input.getShape();

  if (inputShape.size() < 2) {
    // Si la entrada ya es 1D (además del batch) o menos, no se puede aplanar más.
    // En un flujo típico de CNN, la entrada será al menos 2D {batch, features}.
    // Devolvemos la entrada tal cual para manejar este caso.
    return input;
  }

  // 2. Calcular la nueva forma de salida.
  const size_t batchSize = inputShape[0];
  size_t flattenedSize = 1;
  for (size_t i = 1; i < inputShape.size(); ++i) {
    flattenedSize *= inputShape[i];
  }
  const std::vector<size_t> outputShape = {batchSize, flattenedSize};

  // 3. Crear un nuevo tensor de salida y copiar los datos.
  // Es necesario crear un nuevo tensor porque la disposición de la memoria cambia.
  Tensor output(outputShape);

  // NOTA: Esta implementación está optimizada para entrada 4D (B, C, H, W).
  // Es la más común en CNNs para imágenes.
#pragma omp parallel for
  for (size_t b = 0; b < batchSize; ++b) {
    Tensor sample = input.slice(b, 1); // Crea una vista de una muestra.
    size_t flat_idx = 0;
    // Itera sobre el volumen 3D de la muestra y lo copia a la fila de salida.
    for (size_t c = 0; c < sample.getShape()[1]; ++c) {
      for (size_t h = 0; h < sample.getShape()[2]; ++h) {
        for (size_t w = 0; w < sample.getShape()[3]; ++w) {
          // Usamos operator() para acceso seguro a vistas y para la copia.
          output(b, flat_idx) = sample(0, c, h, w);
          flat_idx++;
        }
      }
    }
  }

  return output;
}

/**
 * @brief Realiza el paso hacia atrás, restaurando la forma del gradiente.
 */
Tensor Flatten::backward(const Tensor &outputGradient) {
  // El propósito del backward de Flatten es simplemente una operación de "reshape".
  // El gradiente entrante es plano {batch, flattened_features}, y debe salir con
  // la forma que tenía la entrada original de la capa {batch, C, H, W}.
  Tensor inputGradient(this->inputShape); // Crea un tensor con la forma original.

  const size_t batchSize = this->inputShape[0];
  // const size_t flattenedSize = outputGradient.getShape()[1]; // No es necesario

  // NOTA: Esta implementación también está optimizada para la forma 4D original.
#pragma omp parallel for
  for (size_t b = 0; b < batchSize; ++b) {
    size_t flat_idx = 0;
    // Itera sobre la forma de destino (4D) y extrae valores del gradiente plano.
    for (size_t c = 0; c < this->inputShape[1]; ++c) {
      for (size_t h = 0; h < this->inputShape[2]; ++h) {
        for (size_t w = 0; w < this->inputShape[3]; ++w) {
          inputGradient(b, c, h, w) = outputGradient(b, flat_idx);
          flat_idx++;
        }
      }
    }
  }

  return inputGradient;
}
