#include "capa.hpp"
#include <omp.h>
#include <stdexcept>
using namespace std;

Capa::Capa(int numNeuronas, int numEntradasPorNeurona, const string &activacion, double dropout_rate = 0.0)
    : tipoActivacionCapa(activacion), dropoutRate(dropout_rate) {
  for (int i = 0; i < numNeuronas; ++i) {
    neuronas.emplace_back(numEntradasPorNeurona, activacion);
  }
  ultimasSalidasCapa.resize(numNeuronas);
  if (dropoutRate > 0) {
    dropoutMask.resize(numNeuronas);
  }
}

vector<double> Capa::calcularSalidas(const vector<double> &entradas, bool esEntrenamiento) {
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

  if (dropoutRate > 0.0 && esEntrenamiento) {

#pragma omp parallel
    {
      mt19937 gen_local(random_device{}() + omp_get_thread_num());
      uniform_real_distribution<> dis(0.0, 1.0);

#pragma omp for
      for (size_t i = 0; i < neuronas.size(); ++i) {
        if (dis(gen_local) < dropoutRate) {
          ultimasSalidasCapa[i] = 0.0;
          dropoutMask[i] = false;
        } else {
          ultimasSalidasCapa[i] *= (1.0 / (1.0 - dropoutRate));
          dropoutMask[i] = true;
        }
      }
    }
  }

  return ultimasSalidasCapa;
}

int Capa::obtenerNumNeuronas() const { return neuronas.size(); }

const vector<double> &Capa::obtenerSalidas() const { return ultimasSalidasCapa; }
