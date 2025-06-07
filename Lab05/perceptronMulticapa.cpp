#include "perceptronMulticapa.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace std;

PerceptronMulticapa::PerceptronMulticapa(const vector<int> &neuronasPorCapaConfig,
                                         const vector<string> &funcionesActivacionConfig, double tasaAprendizajeInicial)
    : tasaAprendizaje(tasaAprendizajeInicial), configuracionNeuronasPorCapa(neuronasPorCapaConfig) {

  if (neuronasPorCapaConfig.size() < 2) {
    throw invalid_argument("La red debe tener al menos una capa de entrada y una de salida.");
  }
  if (funcionesActivacionConfig.size() != neuronasPorCapaConfig.size() - 1) {
    throw invalid_argument(
        "El numero de funciones de activacion debe ser igual al numero de capas de procesamiento (total_capas - 1).");
  }

  // Crear capas de procesamiento (ocultas y de salida)
  // neuronasPorCapaConfig[0] es el tamaño de entrada
  // neuronasPorCapaConfig[i+1] es el número de neuronas en la capa i
  // neuronasPorCapaConfig[i] son las entradas para las neuronas en la capa i
  for (size_t i = 0; i < funcionesActivacionConfig.size(); ++i) {
    int numNeuronasEnCapaActual = neuronasPorCapaConfig[i + 1];
    int numEntradasParaNeuronasCapaActual = neuronasPorCapaConfig[i];
    string funcActivacionCapaActual = funcionesActivacionConfig[i];
    capas.emplace_back(numNeuronasEnCapaActual, numEntradasParaNeuronasCapaActual, funcActivacionCapaActual);
  }
}

vector<double> PerceptronMulticapa::propagacionAdelante(const vector<double> &entradas) {
  if (entradas.size() != configuracionNeuronasPorCapa[0]) {
    throw invalid_argument("Tamano de entrada no coincide con la configuracion de la capa de entrada.");
  }

  vector<double> currentInputs = entradas;
  for (size_t i = 0; i < capas.size(); ++i) {
    currentInputs = capas[i].calcularSalidas(currentInputs);
  }
  return currentInputs; // Salidas de la ultima capa
}

void PerceptronMulticapa::retropropagacion(const vector<double> &entradasMuestra, const vector<double> &salidasEsperadas) {
  // forward
  propagacionAdelante(entradasMuestra); // salidas y entradaNeta actualizadas

  // deltas para la capa de salida
  Capa &capaSalida = capas.back();
  if (salidasEsperadas.size() != capaSalida.obtenerNumNeuronas()) {
    throw runtime_error("Desajuste en el tamanio de las salidas esperadas y el numero de neuronas en la capa de salida.");
  }

  for (int k = 0; k < capaSalida.obtenerNumNeuronas(); ++k) {
    Neurona &neuronaK = capaSalida.neuronas[k];
    if (capaSalida.tipoActivacionCapa == "softmax") {
      neuronaK.delta = neuronaK.salida - salidasEsperadas[k];
    } else {
      double error = neuronaK.salida - salidasEsperadas[k];
      neuronaK.delta = error * neuronaK.calcularDerivadaActivacionSalida();
    }
  }

  // deltas para las capas ocultas (iterando hacia atras)
  for (int l = capas.size() - 2; l >= 0; --l) { // desde la penultima capa hasta la primera capa oculta
    Capa &capaActualOculta = capas[l];
    Capa &capaSiguiente = capas[l + 1];

    for (int j = 0; j < capaActualOculta.obtenerNumNeuronas(); ++j) {
      Neurona &neuronaJ = capaActualOculta.neuronas[j];
      double errorPropagado = 0.0;
      for (int k = 0; k < capaSiguiente.obtenerNumNeuronas(); ++k) {
        errorPropagado += capaSiguiente.neuronas[k].pesos[j] * capaSiguiente.neuronas[k].delta;
      }
      neuronaJ.delta = errorPropagado * neuronaJ.calcularDerivadaActivacionSalida();
    }
  }

  // actualizar pesos y sesgos para todas las capas
  for (size_t l = 0; l < capas.size(); ++l) { // it sobre cada capa
    const vector<double> &entradasAEstaCapa = (l == 0) ? entradasMuestra : capas[l - 1].obtenerSalidas();

    if (capas[l].neuronas.empty())
      continue;
    if (!capas[l].neuronas[0].pesos.empty() && entradasAEstaCapa.size() != capas[l].neuronas[0].pesos.size()) {
      throw runtime_error("Desajuste en el tamaño de entrada para la actualizacion de pesos de la capa");
    }

    for (int j = 0; j < capas[l].obtenerNumNeuronas(); ++j) { // it sobre cada neurona
      Neurona &neuronaJ = capas[l].neuronas[j];
      for (size_t i = 0; i < neuronaJ.pesos.size(); ++i) { // it sobre cada peso de la neurona
        neuronaJ.pesos[i] -= tasaAprendizaje * neuronaJ.delta * entradasAEstaCapa[i];
      }
      neuronaJ.sesgo -= tasaAprendizaje * neuronaJ.delta;
    }
  }
}

void PerceptronMulticapa::entrenar(const vector<vector<double>> &entradasEntrenamiento,
                                   const vector<vector<double>> &salidasEntrenamiento, int epocas) {
  if (entradasEntrenamiento.size() != salidasEntrenamiento.size()) {
    throw invalid_argument("El numero de muestras de entrada y salida debe ser el mismo para el entrenamiento.");
  }
  if (entradasEntrenamiento.empty()) {
    cout << "No hay datos de entrenamiento." << endl;
    return;
  }

  // reset historiales
  historialPerdida.clear();
  historialPrecision.clear();

  // timepo inicio del entrenamiento
  auto tiempoInicio = chrono::steady_clock::now();

  for (int epoca = 0; epoca < epocas; ++epoca) {
    double perdidaTotalEpoca = 0.0;
    int prediccionesCorrectasEpoca = 0;

    // it sobre cada muestra de entrenamiento
    for (size_t i = 0; i < entradasEntrenamiento.size(); ++i) {
      // salidas actuales - muestra actual
      vector<double> salidasActuales = propagacionAdelante(entradasEntrenamiento[i]);

      // calcular la perdida Cross-Entropy para esta muestra
      double perdidaMuestra = 0.0;
      for (size_t k = 0; k < salidasActuales.size(); ++k) {
        if (salidasEntrenamiento[i][k] == 1.0) {
          perdidaMuestra -= log(salidasActuales[k] + 1e-9);
        }
      }
      perdidaTotalEpoca += perdidaMuestra;

      // determinar si la prediccion fue correcta para esta muestra
      auto itPredicho = max_element(salidasActuales.begin(), salidasActuales.end());
      int digitoPredicho = distance(salidasActuales.begin(), itPredicho);

      auto itEsperado = max_element(salidasEntrenamiento[i].begin(), salidasEntrenamiento[i].end());
      int digitoEsperado = distance(salidasEntrenamiento[i].begin(), itEsperado);

      if (digitoPredicho == digitoEsperado) {
        prediccionesCorrectasEpoca++;
      }

      // retropropagacion
      retropropagacion(entradasEntrenamiento[i], salidasEntrenamiento[i]);
    }

    // calc perdida promedio y precision
    double perdidaPromedioEpoca = perdidaTotalEpoca / entradasEntrenamiento.size();
    double precisionEpoca = static_cast<double>(prediccionesCorrectasEpoca) / entradasEntrenamiento.size();

    // Almacenar en los historiales
    historialPerdida.push_back(perdidaPromedioEpoca);
    historialPrecision.push_back(precisionEpoca);

    // infor por cada epoca
    cout << "Epoca " << epoca + 1 << "/" << epocas << " - Perdida (Cross-Entropy): " << perdidaPromedioEpoca
         << " - Precision: " << precisionEpoca * 100.0 << "%" << endl;
  }

  // tiempo de fin del entrenamiento
  auto tiempoFin = chrono::steady_clock::now();

  // tiempo total en segundos
  auto duracion = chrono::duration_cast<chrono::seconds>(tiempoFin - tiempoInicio);
  cout << "Tiempo de entrenamiento: " << duracion.count() << " segundos." << endl;
}

vector<double> PerceptronMulticapa::predecir(const vector<double> &entrada) { return propagacionAdelante(entrada); }

void PerceptronMulticapa::guardarPesos(const string &nombreArchivo) const {
  ofstream archivoSalida(nombreArchivo);
  if (!archivoSalida.is_open()) {
    cerr << "Error: No se pudo abrir el archivo para guardar pesos: " << nombreArchivo << endl;
    return;
  }

  // alta precision para los doubles
  archivoSalida << fixed << setprecision(18);

  // Guardar la configuracion de neuronas por capa (incluye la capa de entrada)
  archivoSalida << configuracionNeuronasPorCapa.size() << endl;
  for (size_t i = 0; i < configuracionNeuronasPorCapa.size(); ++i) {
    archivoSalida << configuracionNeuronasPorCapa[i] << (i == configuracionNeuronasPorCapa.size() - 1 ? "" : " ");
  }
  archivoSalida << endl;

  // Guardar los tipos de funcion de activacion de las capas de procesamiento
  archivoSalida << capas.size() << endl; // Numero de capas de procesamiento (ocultas + salida)
  for (const auto &capa : this->capas) {
    archivoSalida << capa.tipoActivacionCapa << endl;
  }

  // Guardar los sesgos y pesos de cada neurona en cada capa de procesamiento
  for (const auto &capa : this->capas) {
    for (const auto &neurona : capa.neuronas) {
      archivoSalida << neurona.sesgo << endl;
      archivoSalida << neurona.pesos.size() << endl;
      for (size_t i = 0; i < neurona.pesos.size(); ++i) {
        archivoSalida << neurona.pesos[i] << (i == neurona.pesos.size() - 1 ? "" : " ");
      }
      archivoSalida << endl;
    }
  }

  archivoSalida.close();
  cout << "Pesos del modelo guardados correctamente en: " << nombreArchivo << endl;
}

void PerceptronMulticapa::cargarPesos(const string &nombreArchivo) {
  ifstream archivoEntrada(nombreArchivo);
  if (!archivoEntrada.is_open()) {
    cerr << "Error: No se pudo abrir el archivo para cargar pesos: " << nombreArchivo << endl;
    return;
  }

  // Validar configuracionNeuronasPorCapa
  size_t numElementosConfigGuardada;
  archivoEntrada >> numElementosConfigGuardada;
  if (archivoEntrada.fail() || numElementosConfigGuardada != this->configuracionNeuronasPorCapa.size()) {
    cerr << "Error al cargar pesos: Incompatibilidad en el numero de elementos de configuracionNeuronasPorCapa." << endl;
    cerr << "Esperado: " << this->configuracionNeuronasPorCapa.size() << ", Archivo: " << numElementosConfigGuardada << endl;
    archivoEntrada.close();
    return;
  }
  for (size_t i = 0; i < this->configuracionNeuronasPorCapa.size(); ++i) {
    int neuronasGuardadasEnCapa;
    archivoEntrada >> neuronasGuardadasEnCapa;
    if (archivoEntrada.fail() || neuronasGuardadasEnCapa != this->configuracionNeuronasPorCapa[i]) {
      cerr << "Error al cargar pesos: Incompatibilidad en el numero de neuronas para la definicion de capa " << i << "."
           << endl;
      cerr << "Esperado: " << this->configuracionNeuronasPorCapa[i] << ", Archivo: " << neuronasGuardadasEnCapa << endl;
      archivoEntrada.close();
      return;
    }
  }

  // Validar tipos de funcion de activacion
  size_t numCapasProcesamientoGuardadas;
  archivoEntrada >> numCapasProcesamientoGuardadas;
  if (archivoEntrada.fail() || numCapasProcesamientoGuardadas != this->capas.size()) {
    cerr << "Error al cargar pesos: Incompatibilidad en el numero de capas de procesamiento." << endl;
    cerr << "Esperado: " << this->capas.size() << ", Archivo: " << numCapasProcesamientoGuardadas << endl;
    archivoEntrada.close();
    return;
  }
  for (size_t i = 0; i < this->capas.size(); ++i) {
    string tipoActivacionGuardado;
    archivoEntrada >> tipoActivacionGuardado; // Lee la palabra (ej. "relu", "softmax")
    if (archivoEntrada.fail() || tipoActivacionGuardado != this->capas[i].tipoActivacionCapa) {
      cerr << "Error al cargar pesos: Incompatibilidad en el tipo de activacion para la capa de procesamiento " << i << "."
           << endl;
      cerr << "Esperado: " << this->capas[i].tipoActivacionCapa << ", Archivo: " << tipoActivacionGuardado << endl;
      archivoEntrada.close();
      return;
    }
  }

  // --- Cargar los sesgos y pesos ---

  for (auto &capa : this->capas) {
    for (auto &neurona : capa.neuronas) {
      archivoEntrada >> neurona.sesgo;
      if (archivoEntrada.fail()) {
        cerr << "Error al leer sesgo del archivo." << endl;
        archivoEntrada.close();
        return;
      }

      size_t numPesosGuardados;
      archivoEntrada >> numPesosGuardados;
      if (archivoEntrada.fail() || numPesosGuardados != neurona.pesos.size()) {
        cerr << "Error al cargar pesos: Incompatibilidad en el numero de pesos para una neurona." << endl;
        cerr << "Esperado: " << neurona.pesos.size() << ", Archivo: " << numPesosGuardados << endl;
        archivoEntrada.close();
        return;
      }
      for (size_t i = 0; i < neurona.pesos.size(); ++i) {
        archivoEntrada >> neurona.pesos[i];
        if (archivoEntrada.fail()) {
          cerr << "Error al leer un peso del archivo." << endl;
          archivoEntrada.close();
          return;
        }
      }
    }
  }

  archivoEntrada.close();
  cout << "Pesos del modelo cargados correctamente desde: " << nombreArchivo << endl;
}

void PerceptronMulticapa::guardarHistorialEntrenamiento(const string &nombreArchivo) const {
  if (historialPerdida.size() != historialPrecision.size()) {
    cerr << "Error: Los historiales de perdida y precision tienen tamanos inconsistentes. No se guardara el historial." << endl;
    return;
  }
  if (historialPerdida.empty()) {
    cout << "Informacion: El historial de entrenamiento esta vacio. No hay nada que guardar." << endl;
    return;
  }

  ofstream archivoSalida(nombreArchivo);
  if (!archivoSalida.is_open()) {
    cerr << "Error: No se pudo abrir el archivo para guardar el historial de entrenamiento: " << nombreArchivo << endl;
    return;
  }

  archivoSalida << fixed << setprecision(10);

  archivoSalida << "Epoca Perdida Precision" << endl;

  for (size_t i = 0; i < historialPerdida.size(); ++i) {
    archivoSalida << (i + 1) // Numero de epoca (comenzando en 1)
                  << " " << historialPerdida[i] << " " << historialPrecision[i] << endl;
  }

  archivoSalida.close();
  cout << "Historial de entrenamiento guardado correctamente en: " << nombreArchivo << endl;
}
