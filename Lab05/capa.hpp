#ifndef CAPA_HPP
#define CAPA_HPP

#include "neurona.hpp"
#include <string>
#include <vector>
using namespace std;

class Capa {
public:
  vector<Neurona> neuronas;
  string tipoActivacionCapa;
  vector<double> ultimasSalidasCapa; // Salidas de la capa en la ultima iteracion

  // Constructor
  Capa(int numNeuronas, int numEntradasPorNeurona, const string &activacion);

  // Calcular las salidas de la capa dada una entrada
  vector<double> calcularSalidas(const vector<double> &entradas);

  int obtenerNumNeuronas() const;

  const vector<double> &obtenerSalidas() const;
};

#endif
