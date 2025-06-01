#include "capa.hpp"
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
    for (size_t i = 0; i < neuronas.size(); ++i) {
      neuronas[i].calcularEntradaNeta(entradas);
      logits[i] = neuronas[i].entradaNeta;
    }

    vector<double> salidasSoftmax = Neurona::softmax(logits);

    for (size_t i = 0; i < neuronas.size(); ++i) {
      neuronas[i].salida = salidasSoftmax[i]; // probabilidad final de softmax como salida de la neurona
      ultimasSalidasCapa[i] = neuronas[i].salida;
    }
  } else {
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
