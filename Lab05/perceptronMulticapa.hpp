
#ifndef PERCEPTRONMULTICAPA_HPP
#define PERCEPTRONMULTICAPA_HPP

#include "capa.hpp"
#include <string>
#include <vector>
using namespace std;

class PerceptronMulticapa {
public:
  vector<Capa> capas;
  double tasaAprendizaje;
  vector<int> configuracionNeuronasPorCapa;
  vector<double> historialPerdida;
  vector<double> historialPrecision;

  // Constructor
  PerceptronMulticapa(const vector<int> &neuronasPorCapaConfig, const vector<string> &funcionesActivacionConfig,
                      double tasaAprendizajeInicial);

  // forward
  vector<double> propagacionAdelante(const vector<double> &entradas);

  // backpropagation
  void retropropagacion(const vector<double> &entradasMuestra, const vector<double> &salidasEsperadas);

  // Entrenamiento
  void entrenar(const vector<vector<double>> &entradasEntrenamiento, const vector<vector<double>> &salidasEntrenamiento,
                int epocas);

  vector<double> predecir(const vector<double> &entrada);

  // guardar pesos
  void guardarPesos(const string &nombreArchivo) const;
  void cargarPesos(const string &nombreArchivo);
  // guardar precision y error
  void guardarHistorialEntrenamiento(const string &nombreArchivo) const;
};

#endif
