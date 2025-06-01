#ifndef NEURONA_HPP
#define NEURONA_HPP

#include <string>
#include <vector>
using namespace std;

class Neurona {
public:
  vector<double> pesos;
  double sesgo;
  double entradaNeta; // suma ponderada + sesgo
  double salida;
  double delta; // error para retropropagacion

  string tipoFuncionActivacion;

  // Constructor de la neurona
  Neurona(int numEntradas, const string &activacion);

  // funciones de activacion
  static double sigmoidea(double x);
  static double relu(double x);
  static double tanhFunc(double x);

  // derivadas de las funciones de activacion
  static double derivadaSigmoidea(double xActivado);
  static double derivadaRelu(double xActivado);
  static double derivadaTanh(double xActivado);

  // softmax para capa de salida
  static vector<double> softmax(const vector<double> &logits);

  void calcularEntradaNeta(const vector<double> &entradas);

  void aplicarActivacion();

  double calcularDerivadaActivacionSalida();
};

#endif
