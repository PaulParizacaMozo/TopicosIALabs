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
  vector<double> historialPerdidaPrueba;
  vector<double> historialPrecisionPrueba;
  string tipoOptimizador;
  double weightDecay;
  bool enEntrenamiento;

  // RMSPROP
  double beta;    // Factor de decaimiento
  double epsilon; // Constante para evitar división por cero

  // ADAM
  double beta1;
  double beta2;
  // double epsilon;
  int t; // Contador de pasos de tiempo para la corrección de sesgo

  // Constructor
  PerceptronMulticapa(const vector<int> &neuronasPorCapaConfig, const vector<string> &funcionesActivacionConfig,
                      const vector<double> &tasasDropoutConfig, double tasaAprendizajeInicial, const string &optimizador,
                      double weight_decay_val = 0.0, double beta = 0.9, double beta1 = 0.9, double beta2 = 0.999,
                      double epsilon = 1e-8);

  // forward
  vector<double> propagacionAdelante(const vector<double> &entradas);

  // backpropagation
  // void retropropagacion(const vector<double> &entradasMuestra, const vector<double> &salidasEsperadas);
  //
  void retropropagacion_acumulada(const vector<double> &entradasMuestra, const vector<double> &salidasEsperadas,
                                  vector<vector<vector<double>>> &acum_grad_pesos, vector<vector<double>> &acum_grad_sesgos);

  // Esta función aplicará los gradientes acumulados
  void aplicar_gradientes_promediados(const vector<vector<vector<double>>> &grad_pesos,
                                      const vector<vector<double>> &grad_sesgos, int tamano_lote);

  // Entrenamiento
  void entrenar(const vector<vector<double>> &entradasEntrenamiento, const vector<vector<double>> &salidasEntrenamiento,
                int epocas, int batch_size, const vector<vector<double>> &entradasPrueba = {},
                const vector<vector<double>> &salidasPrueba = {});

  vector<double> predecir(const vector<double> &entrada);

  // guardar pesos
  void guardarPesos(const string &nombreArchivo) const;
  void cargarPesos(const string &nombreArchivo);
  // guardar precision y error
  void guardarHistorialEntrenamiento(const string &nombreArchivo) const;

  // void actualizarPesos(const vector<double> &entradasMuestra);
  pair<double, double> evaluar(const vector<vector<double>> &entradasPrueba, const vector<vector<double>> &salidasPrueba);
};

#endif
