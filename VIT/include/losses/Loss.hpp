#ifndef LOSS_HPP
#define LOSS_HPP

#include "core/Tensor.hpp"

// Clase base abstracta para todas las funciones de perdida (costo).
// Define la interfaz para calcular el valor de la perdida y su gradiente inicial.
class Loss {
public:
  // Destructor virtual para herencia polimorfica.
  virtual ~Loss() = default;

  // Calcula y devuelve el valor escalar de la perdida.
  // Compara las predicciones del modelo (yPred) con las etiquetas reales (yTrue).
  virtual float calculate(const Tensor &yPred, const Tensor &yTrue) = 0;

  // Calcula el gradiente de la perdida con respecto a las predicciones del modelo.
  // Este es el gradiente inicial que se retropropaga a traves de la red.
  virtual Tensor backward(const Tensor &yPred, const Tensor &yTrue) = 0;
};

#endif // LOSS_HPP
