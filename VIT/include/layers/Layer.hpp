#ifndef LAYER_HPP
#define LAYER_HPP

#include "core/Tensor.hpp"
#include <string>
#include <vector>

// Clase base abstracta para todas las capas de la red.
// Define la interfaz comun para el forward, backward y gestion de parametros.
class Layer {
public:
  // Destructor virtual para herencia polimorfica.
  virtual ~Layer() = default;

  // Realiza el paso hacia adelante (forward pass) de la capa.
  virtual Tensor forward(const Tensor &input, bool isTraining) = 0;

  // Retropropaga el gradiente y calcula los gradientes de los parametros.
  virtual Tensor backward(const Tensor &outputGradient) = 0;

  // Devuelve los parametros entrenables de la capa (pesos, biases).
  virtual std::vector<Tensor *> getParameters() { return {}; }

  // Devuelve los gradientes asociados a los parametros entrenables.
  virtual std::vector<Tensor *> getGradients() { return {}; }

  // Devuelve el nombre de la capa (ej. "Dense").
  virtual std::string getName() const = 0;
};

#endif // LAYER_HPP
