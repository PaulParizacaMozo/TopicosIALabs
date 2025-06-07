#include "neurona.hpp"
#include <random>
#include <stdexcept>
using namespace std;

// Constructor
Neurona::Neurona(int numEntradas, const string &activacion)
    : sesgo(0.0), entradaNeta(0.0), salida(0.0), delta(0.0), tipoFuncionActivacion(activacion) {
  random_device rd;
  mt19937 gen(rd());
  // He para ReLU y Xavier/Glorot para sigmoide/tanh
  double limite =
      sqrt(6.0 / (numEntradas + (tipoFuncionActivacion == "relu" ? numEntradas / 2.0 : numEntradas))); // Xavier/He simplificado
  if (numEntradas > 0) {
    limite = sqrt(1.0 / numEntradas);
  } else {
    limite = 0.05;
  }
  uniform_real_distribution<> dis(-limite, limite);

  for (int i = 0; i < numEntradas; ++i) {
    pesos.push_back(dis(gen));
  }
  sesgo = dis(gen);
}

// Funciones de activacion
double Neurona::sigmoidea(double x) { return 1.0 / (1.0 + exp(-x)); }
double Neurona::relu(double x) { return max(0.0, x); }
double Neurona::tanhFunc(double x) { return tanh(x); }

// Derivadas de funciones de activacion
double Neurona::derivadaSigmoidea(double xActivado) { return xActivado * (1.0 - xActivado); }
double Neurona::derivadaRelu(double xActivado) { return (xActivado > 0) ? 1.0 : 0.0; }
double Neurona::derivadaTanh(double xActivado) { return 1.0 - (xActivado * xActivado); }

// Softmax
vector<double> Neurona::softmax(const vector<double> &logits) {
  vector<double> probabilidades;
  if (logits.empty())
    return probabilidades;

  double maxLogit = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
    }
  }

  double sumaExp = 0.0;
  vector<double> exps;
  exps.reserve(logits.size());

  for (double logit : logits) {
    double valorExp = exp(logit - maxLogit);
    exps.push_back(valorExp);
    sumaExp += valorExp;
  }

  if (sumaExp == 0)
    sumaExp = 1e-9;

  for (double valorExp : exps) {
    probabilidades.push_back(valorExp / sumaExp);
  }
  return probabilidades;
}

void Neurona::calcularEntradaNeta(const vector<double> &entradas) {
  if (entradas.size() != pesos.size()) {
    throw invalid_argument("Desajuste entre el numero de entradas y pesos");
  }
  this->entradaNeta = 0.0;
  for (size_t i = 0; i < pesos.size(); ++i) {
    this->entradaNeta += pesos[i] * entradas[i];
  }
  this->entradaNeta += sesgo;
}

void Neurona::aplicarActivacion() {
  if (tipoFuncionActivacion == "sigmoid") {
    this->salida = sigmoidea(this->entradaNeta);
  } else if (tipoFuncionActivacion == "relu") {
    this->salida = relu(this->entradaNeta);
  } else if (tipoFuncionActivacion == "tanh") {
    this->salida = tanhFunc(this->entradaNeta);
  } else if (tipoFuncionActivacion == "linear" || tipoFuncionActivacion == "softmax") {
    this->salida = this->entradaNeta;
  } else {
    throw invalid_argument("Tipo de funcion de activacion desconocido");
  }
}

double Neurona::calcularDerivadaActivacionSalida() {
  if (tipoFuncionActivacion == "sigmoid") {
    return derivadaSigmoidea(this->salida);
  } else if (tipoFuncionActivacion == "relu") {
    // Para ReLU, la derivada depende de la entradaNeta (z), no directamente de la salida (a) si a=0
    return (this->entradaNeta > 0) ? 1.0 : 0.0;
  } else if (tipoFuncionActivacion == "tanh") {
    return derivadaTanh(this->salida);
  } else if (tipoFuncionActivacion == "linear" || tipoFuncionActivacion == "softmax") {
    return 1.0;
  } else {
    throw invalid_argument("Tipo de funcion de activacion desconocido");
  }
  return 0;
}
