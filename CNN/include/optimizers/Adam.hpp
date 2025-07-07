#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizers/Optimizer.hpp"
#include <vector>

/**
 * @class Adam
 * @brief Implementa el optimizador Adam (Adaptive Moment Estimation).
 *
 * Adam es un algoritmo de optimización adaptativo que combina las ideas de
 * dos extensiones populares de SGD: Momentum y RMSprop.
 *
 * - Mantiene una media móvil exponencial del gradiente (primer momento, `m`).
 * - Mantiene una media móvil exponencial del gradiente al cuadrado (segundo momento, `v`).
 *
 * Esto le permite tener una tasa de aprendizaje adaptativa para cada parámetro
 * individual, lo que a menudo conduce a una convergencia más rápida que SGD.
 */
class Adam : public Optimizer {
public:
  /**
   * @brief Constructor para el optimizador Adam.
   * @param learningRate La tasa de aprendizaje. Típicamente 0.001.
   * @param beta1 Tasa de decaimiento exponencial para la estimación del primer momento. Típicamente 0.9.
   * @param beta2 Tasa de decaimiento exponencial para la estimación del segundo momento. Típicamente 0.999.
   * @param epsilon Un pequeño valor para prevenir la división por cero. Típicamente 1e-8.
   */
  Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8);

  /**
   * @brief Realiza un único paso de actualización de Adam.
   * @details Implementa la lógica completa de Adam, incluyendo la actualización
   *          de los momentos, la corrección de sesgo y la actualización final
   *          de los parámetros.
   * @param parameters Vector de punteros a los parámetros entrenables.
   * @param gradients Vector de punteros a los gradientes correspondientes.
   * @override
   */
  void update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) override;

private:
  // --- Hiperparámetros ---
  float beta1;
  float beta2;
  float epsilon;
  long long t; // Contador de pasos de tiempo, usado para la corrección de sesgo.

  // --- Estado del optimizador ---
  // Almacenan el estado para cada parámetro entrenable en la red.
  std::vector<Tensor> m; ///< Estimación del primer momento (media móvil de gradientes).
  std::vector<Tensor> v; ///< Estimación del segundo momento (media móvil de gradientes al cuadrado).

  // Flag para la inicialización diferida de los tensores de momento.
  bool initialized;
};

#endif // ADAM_HPP
