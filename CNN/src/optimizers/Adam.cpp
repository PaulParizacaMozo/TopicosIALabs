#include "optimizers/Adam.hpp"

#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Constructor que inicializa Adam con sus hiperparámetros.
 */
Adam::Adam(float learningRate, float beta1, float beta2, float epsilon)
    : Optimizer(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0), initialized(false) {}

/**
 * @brief Aplica la regla de actualización de Adam.
 */
void Adam::update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) {
  if (parameters.size() != gradients.size()) {
    throw std::runtime_error("El número de parámetros y gradientes no coincide en Adam::update.");
  }

  // Inicialización diferida: los tensores de momento 'm' y 'v' se crean
  // en la primera llamada a update, una vez que conocemos la forma de los parámetros.
  if (!initialized) {
    m.reserve(parameters.size());
    v.reserve(parameters.size());
    for (const auto &param : parameters) {
      // Crea tensores de momento con la misma forma que los parámetros, inicializados a cero.
      m.emplace_back(param->getShape());
      v.emplace_back(param->getShape());
    }
    initialized = true;
  }

  // Incrementar el contador de pasos de tiempo.
  t++;

  for (size_t i = 0; i < parameters.size(); ++i) {
    Tensor *param = parameters[i];
    const Tensor *grad = gradients[i];
    Tensor &m_t = m[i];
    Tensor &v_t = v[i];

    const auto &shape = param->getShape();

    // Pre-calcular los factores de corrección de sesgo
    const float beta1_t = std::pow(beta1, t);
    const float beta2_t = std::pow(beta2, t);

    // Especialización para 2D y 4D
    if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
      for (size_t r = 0; r < shape[0]; ++r) {
        for (size_t c = 0; c < shape[1]; ++c) {
          float g = (*grad)(r, c);

          // 1. Actualizar momentos (medias móviles exponenciales)
          m_t(r, c) = beta1 * m_t(r, c) + (1.0f - beta1) * g;
          v_t(r, c) = beta2 * v_t(r, c) + (1.0f - beta2) * (g * g);

          // 2. Corregir el sesgo de los momentos
          float m_hat = m_t(r, c) / (1.0f - beta1_t);
          float v_hat = v_t(r, c) / (1.0f - beta2_t);

          // 3. Actualizar el parámetro
          (*param)(r, c) -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
      }
    } else if (shape.size() == 4) {
#pragma omp parallel for collapse(4)
      for (size_t d0 = 0; d0 < shape[0]; ++d0) {
        for (size_t d1 = 0; d1 < shape[1]; ++d1) {
          for (size_t d2 = 0; d2 < shape[2]; ++d2) {
            for (size_t d3 = 0; d3 < shape[3]; ++d3) {
              float g = (*grad)(d0, d1, d2, d3);

              m_t(d0, d1, d2, d3) = beta1 * m_t(d0, d1, d2, d3) + (1.0f - beta1) * g;
              v_t(d0, d1, d2, d3) = beta2 * v_t(d0, d1, d2, d3) + (1.0f - beta2) * (g * g);

              float m_hat = m_t(d0, d1, d2, d3) / (1.0f - beta1_t);
              float v_hat = v_t(d0, d1, d2, d3) / (1.0f - beta2_t);

              (*param)(d0, d1, d2, d3) -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
          }
        }
      }
    }
  }
}
