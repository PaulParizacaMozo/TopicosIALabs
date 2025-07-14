#ifndef CROSSENTROPY_HPP
#define CROSSENTROPY_HPP

#include "losses/Loss.hpp"

// Declaracion adelantada de la funcion softmax.
Tensor softmax(const Tensor &logits, int axis);

// Implementa la funcion de perdida de Entropia Cruzada Categórica.
// Combina la activacion Softmax y la perdida de Entropia Cruzada.
// Este enfoque es numericamente mas estable y el gradiente se simplifica a:
// (softmax_output - true_labels).
class CrossEntropy : public Loss {
public:
  // Constructor.
  CrossEntropy() = default;

  // Calcula la perdida de entropia cruzada.
  // Primero aplica Softmax a los logits (yPred) y luego calcula la perdida.
  float calculate(const Tensor &yPred, const Tensor &yTrue) override;

  // Calcula el gradiente inicial para la retropropagacion.
  // Reutiliza las probabilidades calculadas en el paso 'calculate'.
  Tensor backward(const Tensor &yPred, const Tensor &yTrue) override;

private:
  // Almacena las probabilidades de Softmax calculadas en 'calculate'
  // para ser reutilizadas en 'backward'.
  Tensor softmaxOutput;
};

#endif // CROSSENTROPY_HPP
