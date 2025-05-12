// main.cpp
#include "perceptron.hpp"
#include <iostream>
#include <vector>

using namespace std;

int main() {
  vector<vector<float>> entradas = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  float tasa_aprendizaje = 0.1;
  int max_iter = 1000;
  vector<vector<float>> pruebas = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

  vector<int> salidas_and = {0, 0, 0, 1}; // Funcion AND

  // Entrenamiento y pruebas para AND
  cout << "\n=== Entrenamiento para la funcion AND con escalon ===\n";
  Perceptron perceptron_and(2, tasa_aprendizaje);
  perceptron_and.entrenar(entradas, salidas_and, max_iter);

  // Probando nuevas entradas para AND
  cout << "\nProbando nuevas entradas para AND (Escalon):\n";
  for (const auto &entrada : pruebas) {
    cout << "Entrada: [" << entrada[0] << ", " << entrada[1] << "] -> Prediccion: " << perceptron_and.predecir(entrada) << "\n";
  }

  // Entrenamiento y pruebas para OR
  vector<int> salidas_or = {0, 1, 1, 1}; // Funcion OR
  cout << "\n=== Entrenamiento para la funcion OR con escalon ===\n";
  Perceptron perceptron_or(2, tasa_aprendizaje);
  perceptron_or.entrenar(entradas, salidas_or, max_iter);

  // Probando nuevas entradas para OR
  cout << "\nProbando nuevas entradas para OR (Escalon):\n";
  for (const auto &entrada : pruebas) {
    cout << "Entrada: [" << entrada[0] << ", " << entrada[1] << "] -> Prediccion: " << perceptron_or.predecir(entrada) << "\n";
  }

  return 0;
}
