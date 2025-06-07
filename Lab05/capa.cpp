#include "capa.hpp"
#include <omp.h>
#include <stdexcept>
using namespace std;

Capa::Capa(int numNeuronas, int numEntradasPorNeurona, const string &activacion) : tipoActivacionCapa(activacion) {
  for (int i = 0; i < numNeuronas; ++i) {
    neuronas.emplace_back(numEntradasPorNeurona, activacion);
  }
  ultimasSalidasCapa.resize(numNeuronas);
}

vector<double> Capa::calcularSalidas(const vector<double> &entradas) {
  if (neuronas.empty()) {
    throw runtime_error("La capa no tiene neuronas.");
  }
  if (!neuronas.empty() && !neuronas[0].pesos.empty() && entradas.size() != neuronas[0].pesos.size()) {
    throw runtime_error("Desajuste de tamanio de entrada y peso");
  }

  if (tipoActivacionCapa == "softmax") {
    vector<double> logits(neuronas.size());

// Paralelización del cálculo de entrada neta y logits
#pragma omp parallel for
    for (size_t i = 0; i < neuronas.size(); ++i) {
      neuronas[i].calcularEntradaNeta(entradas);
      logits[i] = neuronas[i].entradaNeta;
    }

    // Calculamos las salidas softmax
    vector<double> salidasSoftmax = Neurona::softmax(logits);

// Paralelización de la actualización de salidas de neuronas
#pragma omp parallel for
    for (size_t i = 0; i < neuronas.size(); ++i) {
      neuronas[i].salida = salidasSoftmax[i];
      ultimasSalidasCapa[i] = neuronas[i].salida;
    }

  } else {
// Paralelización de la actualización de entrada neta y activación
#pragma omp parallel for
    for (size_t i = 0; i < neuronas.size(); ++i) {
      neuronas[i].calcularEntradaNeta(entradas);
      neuronas[i].aplicarActivacion(); // aplica sigmoid, relu, tanh
      ultimasSalidasCapa[i] = neuronas[i].salida;
    }
  }

  return ultimasSalidasCapa;
}

int Capa::obtenerNumNeuronas() const { return neuronas.size(); }

const vector<double> &Capa::obtenerSalidas() const { return ultimasSalidasCapa; }
