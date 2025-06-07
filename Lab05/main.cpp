#include "mnist_utils.hpp"
#include "perceptronMulticapa.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

void mostrarImagenConsola(const vector<double> &entrada) {
  int size = 28;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (entrada[i * size + j] > 0.5)
        cout << "#";
      else
        cout << " ";
    }
    cout << endl;
  }
}
void mostrarPrediccion(const vector<double> &entrada, const vector<double> &prediccion, const vector<double> &esperado) {
  cout << "Imagen de entrada (28x28):" << endl;
  mostrarImagenConsola(entrada);

  cout << "Salida Esperada (one-hot): ";
  for (double val : esperado)
    cout << val << " ";
  auto itEsperado = max_element(esperado.begin(), esperado.end());
  cout << " -> Digito: " << distance(esperado.begin(), itEsperado) << endl;

  cout << "Prediccion (softmax):     ";
  for (double prob : prediccion)
    printf("%.4f ", prob);
  auto itPredicho = max_element(prediccion.begin(), prediccion.end());
  cout << " -> Digito: " << distance(prediccion.begin(), itPredicho) << endl;

  if (distance(esperado.begin(), itEsperado) == distance(prediccion.begin(), itPredicho)) {
    cout << "Resultado: CORRECTO" << endl;
  } else {
    cout << "Resultado: INCORRECTO" << endl;
  }
  cout << "------------------------------------------" << endl;
}
int main() {
  // estructura de la red
  // Capa de entrada: 784 neuronas (pixeles 28x28)
  // Capa oculta: 128 neuronas con activacion ReLU
  // Capa oculta: 64 neuronas con activacion ReLU
  // Capa de salida: 10 neuronas (digitos 0-9) con activacion Softmax
  vector<int> neuronasPorCapa = {784, 128, 64, 10};
  vector<string> funcionesActivacion = {"relu", "relu", "softmax"};

  // Parametros de entrenamiento
  double tasaAprendizaje = 0.01;
  int epocas = 20;
  int numMuestrasEntrenamiento = 60000;

  // Cargar datos de entrenamiento
  cout << "Cargando datos de entrenamiento..." << endl;
  MNISTData datosEntrenamiento = cargarDatosCSV("mnist_train.csv", numMuestrasEntrenamiento);

  if (datosEntrenamiento.entradas.empty()) {
    cerr << "No se pudieron cargar los datos de entrenamiento. Terminando." << endl;
    return 1;
  }

  cout << "Dimensiones de entrada[0]: " << datosEntrenamiento.entradas[0].size() << endl;
  cout << "Dimensiones de salidaEsperada[0]: " << datosEntrenamiento.salidasEsperadas[0].size() << endl;

  // crear la red neuronal
  cout << "Creando la red neuronal..." << endl;
  PerceptronMulticapa red(neuronasPorCapa, funcionesActivacion, tasaAprendizaje);
  cout << "Red neuronal creada." << endl;

  // entrenamiento de la red
  cout << "Iniciando entrenamiento..." << endl;
  red.entrenar(datosEntrenamiento.entradas, datosEntrenamiento.salidasEsperadas, epocas);
  cout << "Entrenamiento completado." << endl;
  // guardar pesos y historial de entrenamiento
  const string nombreArchivoPesos = "modelo_mnist_pesos.txt";
  red.guardarPesos(nombreArchivoPesos);
  const string nombreArchivoHistorial = "historial_entrenamiento.txt";
  red.guardarHistorialEntrenamiento(nombreArchivoHistorial);

  cout << "\n--- Historial del Entrenamiento ---" << endl;
  if (red.historialPerdida.size() == red.historialPrecision.size()) {
    for (size_t i = 0; i < red.historialPerdida.size(); ++i) {
      cout << "Epoca " << i + 1 << ": "
           << "Perdida = " << red.historialPerdida[i] << ", Precision = " << red.historialPrecision[i] * 100.0 << "%" << endl;
    }
  }
  cout << "----------------------------------" << endl;

  // cargar datos de prueba y evaluar la precision general
  cout << "\nCargando datos de prueba para evaluacion..." << endl;
  MNISTData datosPrueba = cargarDatosCSV("mnist_test.csv", 10000); // Cargar 10000 muestras de prueba
  if (!datosPrueba.entradas.empty()) {
    int correctas = 0;
    for (size_t i = 0; i < datosPrueba.entradas.size(); ++i) {
      vector<double> prediccion = red.predecir(datosPrueba.entradas[i]);

      auto itEsperado = max_element(datosPrueba.salidasEsperadas[i].begin(), datosPrueba.salidasEsperadas[i].end());
      int digitoEsperado = distance(datosPrueba.salidasEsperadas[i].begin(), itEsperado);

      auto itPredicho = max_element(prediccion.begin(), prediccion.end());
      int digitoPredicho = distance(prediccion.begin(), itPredicho);

      if (digitoPredicho == digitoEsperado) {
        correctas++;
      }
      // if (i < 3) {
      //   mostrarPrediccion(datosPrueba.entradas[i], prediccion, datosPrueba.salidasEsperadas[i]);
      // }

      if ((digitoEsperado != digitoPredicho) && i < 10) {
        mostrarPrediccion(datosPrueba.entradas[i], prediccion, datosPrueba.salidasEsperadas[i]);
      }
    }
    double precision = static_cast<double>(correctas) / datosPrueba.entradas.size() * 100.0;
    cout << "\nPrecision en el conjunto de prueba (" << datosPrueba.entradas.size() << " muestras): " << precision << "% ("
         << correctas << "/" << datosPrueba.entradas.size() << ")" << endl;
  }

  // --- cargar y evaluar modelo cargado ---
  cout << "\n--- Probando el modelo cargado ---" << endl;

  PerceptronMulticapa redCargada(neuronasPorCapa, funcionesActivacion, tasaAprendizaje);
  redCargada.cargarPesos(nombreArchivoPesos);

  cout << "\nEvaluando la RED CARGADA con datos de prueba..." << endl;
  if (!datosPrueba.entradas.empty()) {
    int correctasCargada = 0;
    for (size_t i = 0; i < datosPrueba.entradas.size(); ++i) {
      vector<double> prediccion = redCargada.predecir(datosPrueba.entradas[i]);

      auto itEsperado = max_element(datosPrueba.salidasEsperadas[i].begin(), datosPrueba.salidasEsperadas[i].end());
      int digitoEsperado = distance(datosPrueba.salidasEsperadas[i].begin(), itEsperado);

      auto itPredicho = max_element(prediccion.begin(), prediccion.end());
      int digitoPredicho = distance(prediccion.begin(), itPredicho);

      if (digitoPredicho == digitoEsperado) {
        correctasCargada++;
      }
      if ((digitoEsperado == digitoPredicho) && i < 5) {
        mostrarPrediccion(datosPrueba.entradas[i], prediccion, datosPrueba.salidasEsperadas[i]);
      }
    }
    double precisionCargada = static_cast<double>(correctasCargada) / datosPrueba.entradas.size() * 100.0;
    cout << "Precision de la RED CARGADA en el conjunto de prueba (" << datosPrueba.entradas.size()
         << " muestras): " << precisionCargada << "% (" << correctasCargada << "/" << datosPrueba.entradas.size() << ")"
         << endl;
  }
  cout << "----------------------------------" << endl;

  return 0;
}
