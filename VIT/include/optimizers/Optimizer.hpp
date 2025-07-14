#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "core/Tensor.hpp"
#include <vector>

// Clase base abstracta para todos los algoritmos de optimizacion.
// Define la interfaz para optimizadores como SGD o Adam, cuya tarea es
// actualizar los parametros de la red usando los gradientes calculados.
class Optimizer {
public:
  // Constructor que define la tasa de aprendizaje.
  explicit Optimizer(float learningRate) : learningRate(learningRate) {}

  // Destructor virtual para herencia polimorfica.
  virtual ~Optimizer() = default;

  // Realiza un unico paso de optimizacion para actualizar los parametros.
  // - parameters: Punteros a los parametros entrenables del modelo.
  // - gradients: Punteros a los gradientes correspondientes a cada parametro.
  virtual void update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) = 0;

protected:
  // Tasa de aprendizaje (learning rate) del algoritmo.
  float learningRate;
};

#endif // OPTIMIZER_HPP
