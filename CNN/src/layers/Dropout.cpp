#include "layers/Dropout.hpp"

#include <random>
#include <stdexcept>

// Incluir OpenMP si está disponible
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Constructor de la capa Dropout.
 */
Dropout::Dropout(float rate) : rate(rate) {
  if (rate < 0.0f || rate >= 1.0f) {
    throw std::invalid_argument("La tasa de Dropout debe estar en el rango [0, 1).");
  }
  // Se pre-calcula el factor de escala para "inverted dropout".
  // Esto evita tener que escalar en el momento de la inferencia.
  this->scale = 1.0f / (1.0f - rate);
}

/**
 * @brief Aplica el dropout durante el entrenamiento.
 */
Tensor Dropout::forward(const Tensor &input, bool isTraining) {
  // Durante la inferencia, el dropout no se aplica. La capa es transparente.
  if (!isTraining) {
    return input;
  }

  // 1. Crear la máscara de dropout.
  this->dropoutMask = Tensor(input.getShape());
  const auto &shape = input.getShape();

  // --- Generación de números aleatorios segura para hilos ---
  // Se crea una semilla base fuera de la región paralela.
  std::random_device rd;
  std::mt19937 seeder(rd());

#pragma omp parallel
  {
    // Cada hilo crea su propio generador de números aleatorios,
    // sembrado con una semilla única para evitar que todos los hilos
    // generen la misma secuencia de números.
    std::mt19937 generator(seeder() + omp_get_thread_num());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    // Se especializa para las formas 2D y 4D más comunes.
    if (shape.size() == 2) {
#pragma omp for collapse(2)
      for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
          dropoutMask(i, j) = (distribution(generator) > this->rate) ? this->scale : 0.0f;
        }
      }
    } else if (shape.size() == 4) {
#pragma omp for collapse(4)
      for (size_t b = 0; b < shape[0]; ++b) {
        for (size_t c = 0; c < shape[1]; ++c) {
          for (size_t h = 0; h < shape[2]; ++h) {
            for (size_t w = 0; w < shape[3]; ++w) {
              dropoutMask(b, c, h, w) = (distribution(generator) > this->rate) ? this->scale : 0.0f;
            }
          }
        }
      }
    }
    // Si la forma no es 2D ni 4D, el `omp for` no se ejecuta y la máscara queda en ceros.
    // Un `else` con una excepción aquí podría ser más explícito.
  } // Fin de la región paralela

  // 2. Aplicar la máscara (multiplicación elemento a elemento).
  // Podríamos unificar los bucles de creación y aplicación de la máscara,
  // pero separarlos puede ser más claro.
  Tensor output(input.getShape());
  if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        output(i, j) = input(i, j) * this->dropoutMask(i, j);
      }
    }
  } else if (shape.size() == 4) {
#pragma omp parallel for collapse(4)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t c = 0; c < shape[1]; ++c) {
        for (size_t h = 0; h < shape[2]; ++h) {
          for (size_t w = 0; w < shape[3]; ++w) {
            output(b, c, h, w) = input(b, c, h, w) * this->dropoutMask(b, c, h, w);
          }
        }
      }
    }
  } else {
    throw std::runtime_error("Dropout::forward solo soporta entradas 2D o 4D.");
  }

  return output;
}

/**
 * @brief Retropropaga el gradiente aplicando la misma máscara.
 */
Tensor Dropout::backward(const Tensor &outputGradient) {
  // La derivada de la operación de dropout es simplemente la propia máscara.
  // Por la regla de la cadena: dE/dX = dE/dY * (dY/dX) = dE/dY * dropoutMask.
  Tensor inputGradient(outputGradient.getShape());
  const auto &shape = outputGradient.getShape();

  if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        inputGradient(i, j) = outputGradient(i, j) * this->dropoutMask(i, j);
      }
    }
  } else if (shape.size() == 4) {
#pragma omp parallel for collapse(4)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t c = 0; c < shape[1]; ++c) {
        for (size_t h = 0; h < shape[2]; ++h) {
          for (size_t w = 0; w < shape[3]; ++w) {
            inputGradient(b, c, h, w) = outputGradient(b, c, h, w) * this->dropoutMask(b, c, h, w);
          }
        }
      }
    }
  } else {
    throw std::runtime_error("Dropout::backward solo soporta entradas 2D o 4D.");
  }

  return inputGradient;
}
