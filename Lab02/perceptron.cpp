// perceptron.cpp
#include "perceptron.hpp"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

using namespace std;

// Constructor
Perceptron::Perceptron(int num_entradas, float tasa) : n_entradas(num_entradas), tasa_aprendizaje(tasa), pesos(num_entradas), bias(0.0f) {
  // Inicializar pesos y bias aleatoriamente entre -1 y 1
  srand(1002);
  for (int i = 0; i < n_entradas; ++i) {
    pesos[i] = (float(rand()) / RAND_MAX) * 2 - 1;
  }
  bias = (float(rand()) / RAND_MAX) * 2 - 1;
}

// Funcion de activacion escalon
int Perceptron::func_escalon(float z) { return (z >= 0) ? 1 : 0; }

// Entrenar el perceptron
void Perceptron::entrenar(const vector<vector<float>> &entradas, const vector<int> &salidas, int max_iter) {
  bool convergido = false;
  cout << "Iniciando entrenamiento con activacion escalon...\n";

  // Inicio crono
  auto inicio = chrono::high_resolution_clock::now();

  for (int iter = 0; iter < max_iter && !convergido; ++iter) {
    convergido = true;
    vector<int> predicciones(entradas.size());

    cout << "\nIteracion " << iter + 1 << ":\n";
    cout << "Pesos: ";
    for (float w : pesos) {
      cout << w << " ";
    }
    cout << ", bias: " << bias << "\n";

    // Procesar cada ejemplo
    for (size_t i = 0; i < entradas.size(); ++i) {
      // Calcular suma ponderada
      float z = bias;
      for (int j = 0; j < n_entradas; ++j) {
        z += pesos[j] * entradas[i][j];
      }
      predicciones[i] = func_escalon(z); // Usar directamente func_escalon
      int error = salidas[i] - predicciones[i];
      // Actualizar pesos y bias si hay error
      if (error != 0) {
        convergido = false;
        for (int j = 0; j < n_entradas; ++j) {
          pesos[j] += tasa_aprendizaje * error * entradas[i][j];
        }
        bias += tasa_aprendizaje * error;
      }
    }

    // Mostrar predicciones
    cout << "Predicciones: ";
    for (int p : predicciones)
      cout << p << " ";
    cout << "\nEsperado:    ";
    for (int s : salidas)
      cout << s << " ";
    cout << "\n";

    if (convergido) {
      cout << "Convergio despues de " << iter + 1 << " iteraciones.\n";
    }
  }

  if (!convergido) {
    cout << "No convergio dentro de " << max_iter << " iteraciones.\n";
  }

  cout << "\nPesos finales: ";
  for (float w : pesos)
    cout << w << " ";
  cout << ", bias: " << bias << "\n";

  // Fin crono
  auto fin = chrono::high_resolution_clock::now();
  auto duracion = chrono::duration_cast<chrono::microseconds>(fin - inicio);
  cout << "\nTiempo de entrenamiento: " << duracion.count() << " microsegundos.\n";
}

// Predecir la salida para una nueva entrada
int Perceptron::predecir(const vector<float> &entrada) {
  if (entrada.size() != static_cast<size_t>(n_entradas)) {
    cout << "Error: La entrada debe tener " << n_entradas << " valores.\n";
    return -1;
  }
  float z = bias;
  for (int i = 0; i < n_entradas; ++i) {
    z += pesos[i] * entrada[i];
  }
  return func_escalon(z);
}
