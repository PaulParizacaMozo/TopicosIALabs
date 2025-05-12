// perceptron.hpp
#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <vector>

class Perceptron {
private:
  int n_entradas;            // N entradas
  float tasa_aprendizaje;    // tasa de aprendizaje
  float bias;                // bias
  std::vector<float> pesos;  // vector pesos
  int func_escalon(float z); // funcion de activacion

public:
  Perceptron(int num_entradas, float tasa = 0.1);

  float activacion(float z);

  void entrenar(const std::vector<std::vector<float>> &entradas, const std::vector<int> &salidas, int max_iter);

  int predecir(const std::vector<float> &entrada);
};

#endif // PERCEPTRON_HPP
