#include "optimizers/SGD.hpp"

#include <stdexcept>

// Incluir OpenMP si está disponible
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Constructor que inicializa el optimizador SGD con una tasa de aprendizaje.
 */
SGD::SGD(float learningRate) : Optimizer(learningRate) {}

/**
 * @brief Aplica la regla de actualización de SGD a un conjunto de parámetros.
 */
void SGD::update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) {
  // Verificación de sanidad: debe haber un gradiente por cada parámetro.
  if (parameters.size() != gradients.size()) {
    throw std::runtime_error("El número de parámetros y gradientes no coincide en SGD::update.");
  }

  // Itera sobre cada par (parámetro, gradiente).
  // Nota: Este bucle externo no se paraleliza porque cada iteración opera
  // en tensores diferentes. La paralelización ocurre dentro de cada tensor.
  for (size_t i = 0; i < parameters.size(); ++i) {
    Tensor *param = parameters[i];
    const Tensor *grad = gradients[i];

    if (param->getSize() != grad->getSize()) {
      throw std::runtime_error("La forma de un parámetro y su gradiente no coincide.");
    }

    const auto &shape = param->getShape();

    // La actualización es una operación elemento a elemento, ideal para paralelizar.
    // Se especializa para las formas más comunes (pesos de Conv2D y Dense).
    if (shape.size() == 2) { // Caso 2D (ej. pesos de Dense)
#pragma omp parallel for collapse(2)
      for (size_t r = 0; r < shape[0]; ++r) {
        for (size_t c = 0; c < shape[1]; ++c) {
          // Regla de actualización: param -= lr * grad
          (*param)(r, c) -= this->learningRate * (*grad)(r, c);
        }
      }
    } else if (shape.size() == 4) { // Caso 4D (ej. pesos de Conv2D)
#pragma omp parallel for collapse(4)
      for (size_t d0 = 0; d0 < shape[0]; ++d0) {
        for (size_t d1 = 0; d1 < shape[1]; ++d1) {
          for (size_t d2 = 0; d2 < shape[2]; ++d2) {
            for (size_t d3 = 0; d3 < shape[3]; ++d3) {
              (*param)(d0, d1, d2, d3) -= this->learningRate * (*grad)(d0, d1, d2, d3);
            }
          }
        }
      }
    } else if (shape.size() == 1) { // Caso 1D (ej. podría ser un bias)
#pragma omp parallel for
      for (size_t d0 = 0; d0 < shape[0]; ++d0) {
        (*param)(d0) -= this->learningRate * (*grad)(d0);
      }
    } else {
      // Es más seguro lanzar un error que fallar silenciosamente.
      throw std::runtime_error("SGD::update solo soporta tensores 1D, 2D o 4D en este momento.");
    }
  }
}
