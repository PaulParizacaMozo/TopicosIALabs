#include "perceptronMulticapa.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stdexcept>
using namespace std;

PerceptronMulticapa::PerceptronMulticapa(const vector<int> &neuronasPorCapaConfig,
                                         const vector<string> &funcionesActivacionConfig,
                                         const vector<double> &tasasDropoutConfig, double tasaAprendizajeInicial,
                                         const string &optimizador, double weight_decay_val, double beta_val, double beta1_val,
                                         double beta2_val, double epsilon_val)
    : tasaAprendizaje(tasaAprendizajeInicial), enEntrenamiento(false), tipoOptimizador(optimizador),
      weightDecay(weight_decay_val), // <-- Guardamos el valor
      beta(beta_val), beta1(beta1_val), beta2(beta2_val), epsilon(epsilon_val), t(0),
      configuracionNeuronasPorCapa(neuronasPorCapaConfig) {

  if (neuronasPorCapaConfig.size() < 2) {
    throw invalid_argument("La red debe tener al menos una capa de entrada y una de salida.");
  }
  if (funcionesActivacionConfig.size() != neuronasPorCapaConfig.size() - 1) {
    throw invalid_argument(
        "El numero de funciones de activacion debe ser igual al numero de capas de procesamiento (total_capas - 1).");
  }
  if (tasasDropoutConfig.size() != neuronasPorCapaConfig.size() - 1) {
    throw invalid_argument("El numero de tasas de dropout debe ser igual al numero de capas de procesamiento.");
  }

  // Crear capas de procesamiento (ocultas y de salida)
  // neuronasPorCapaConfig[0] es el tamaño de entrada
  // neuronasPorCapaConfig[i+1] es el número de neuronas en la capa i
  // neuronasPorCapaConfig[i] son las entradas para las neuronas en la capa i
  for (size_t i = 0; i < funcionesActivacionConfig.size(); ++i) {
    int numNeuronasEnCapaActual = neuronasPorCapaConfig[i + 1];
    int numEntradasParaNeuronasCapaActual = neuronasPorCapaConfig[i];
    string funcActivacionCapaActual = funcionesActivacionConfig[i];
    capas.emplace_back(numNeuronasEnCapaActual, numEntradasParaNeuronasCapaActual, funcActivacionCapaActual,
                       tasasDropoutConfig[i]);
  }
}

vector<double> PerceptronMulticapa::propagacionAdelante(const vector<double> &entradas) {
  if (entradas.size() != configuracionNeuronasPorCapa[0]) {
    throw invalid_argument("Tamano de entrada no coincide con la configuracion de la capa de entrada.");
  }

  vector<double> currentInputs = entradas;
  for (size_t i = 0; i < capas.size(); ++i) {
    currentInputs = capas[i].calcularSalidas(currentInputs, this->enEntrenamiento);
  }
  return currentInputs; // Salidas de la ultima capa
}

void PerceptronMulticapa::entrenar(const vector<vector<double>> &entradasEntrenamiento,
                                   const vector<vector<double>> &salidasEntrenamiento, int epocas, int batch_size,
                                   const vector<vector<double>> &entradasPrueba, const vector<vector<double>> &salidasPrueba) {
  if (entradasEntrenamiento.size() != salidasEntrenamiento.size()) {
    throw invalid_argument("El numero de muestras de entrada y salida debe ser el mismo para el entrenamiento.");
  }
  if (entradasEntrenamiento.empty()) {
    cout << "No hay datos de entrenamiento." << endl;
    return;
  }

  // resetear historiales y contadores
  historialPerdida.clear();
  historialPrecision.clear();
  historialPerdidaPrueba.clear();
  historialPrecisionPrueba.clear();
  this->t = 0;

  size_t num_muestras = entradasEntrenamiento.size();
  auto tiempoInicio = chrono::steady_clock::now();

  for (int epoca = 0; epoca < epocas; ++epoca) {
    this->enEntrenamiento = true;

    // mezclar los datos de entrenamiento para la nueva epoca
    vector<int> indices(num_muestras);
    iota(indices.begin(), indices.end(), 0); // Llena el vector con 0, 1, 2, ...
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(indices.begin(), indices.end(), default_random_engine(seed));

    double perdidaTotalEpoca = 0.0;
    int prediccionesCorrectasEpoca = 0;

    // iterar sobre el dataset en mini-lotes
    for (size_t i = 0; i < num_muestras; i += batch_size) {
      // Inicializar acumuladores de gradientes para el lote
      vector<vector<vector<double>>> gradientes_pesos_lote(capas.size());
      vector<vector<double>> gradientes_sesgos_lote(capas.size());
      for (size_t l = 0; l < capas.size(); ++l) {
        gradientes_pesos_lote[l].resize(capas[l].neuronas.size());
        gradientes_sesgos_lote[l].resize(capas[l].neuronas.size(), 0.0);
        for (size_t j = 0; j < capas[l].neuronas.size(); ++j) {
          gradientes_pesos_lote[l][j].resize(capas[l].neuronas[j].pesos.size(), 0.0);
        }
      }

      size_t fin_lote = min(i + batch_size, num_muestras);
      int tamano_real_lote = fin_lote - i;

      // procesar cada muestra dentro del lote
      for (size_t j = i; j < fin_lote; ++j) {
        int indice_muestra = indices[j];

        // forward
        vector<double> salidasActuales = propagacionAdelante(entradasEntrenamiento[indice_muestra]);

        // acumular metricas de la epoca
        double perdidaMuestra = 0.0;
        for (size_t k = 0; k < salidasActuales.size(); ++k) {
          if (salidasEntrenamiento[indice_muestra][k] == 1.0) {
            perdidaMuestra -= log(salidasActuales[k] + 1e-9);
          }
        }
        perdidaTotalEpoca += perdidaMuestra;

        auto itPredicho = max_element(salidasActuales.begin(), salidasActuales.end());
        auto itEsperado = max_element(salidasEntrenamiento[indice_muestra].begin(), salidasEntrenamiento[indice_muestra].end());
        if (distance(salidasActuales.begin(), itPredicho) ==
            distance(salidasEntrenamiento[indice_muestra].begin(), itEsperado)) {
          prediccionesCorrectasEpoca++;
        }

        // backward
        retropropagacion_acumulada(entradasEntrenamiento[indice_muestra], salidasEntrenamiento[indice_muestra],
                                   gradientes_pesos_lote, gradientes_sesgos_lote);
      }

      // incrementar el contador de pasos y aplicar actualizacion de pesos (una vez por lote)
      this->t++; // contador para Adam
      aplicar_gradientes_promediados(gradientes_pesos_lote, gradientes_sesgos_lote, tamano_real_lote);
    }

    // calcular y mostrar metricas de la epoca
    double perdidaPromedioEpoca = perdidaTotalEpoca / num_muestras;
    double precisionEpoca = static_cast<double>(prediccionesCorrectasEpoca) / num_muestras;
    historialPerdida.push_back(perdidaPromedioEpoca);
    historialPrecision.push_back(precisionEpoca);

    cout << "Epoca " << epoca + 1 << "/" << epocas << " - Perdida (Train): " << fixed << setprecision(4) << perdidaPromedioEpoca
         << " - Precision (Train): " << precisionEpoca * 100.0 << "%";

    if (!entradasPrueba.empty()) {
      pair<double, double> metricasPrueba = evaluar(entradasPrueba, salidasPrueba);
      historialPerdidaPrueba.push_back(metricasPrueba.first);
      historialPrecisionPrueba.push_back(metricasPrueba.second);
      cout << " - Perdida (Test): " << metricasPrueba.first << " - Precision (Test): " << metricasPrueba.second * 100.0 << "%";
    }
    cout << endl;
  }

  this->enEntrenamiento = false;
  auto tiempoFin = chrono::steady_clock::now();
  auto duracion = chrono::duration_cast<chrono::seconds>(tiempoFin - tiempoInicio);
  cout << "Tiempo de entrenamiento: " << duracion.count() << " segundos." << endl;
}

void PerceptronMulticapa::retropropagacion_acumulada(const vector<double> &entradasMuestra,
                                                     const vector<double> &salidasEsperadas,
                                                     vector<vector<vector<double>>> &acum_grad_pesos,
                                                     vector<vector<double>> &acum_grad_sesgos) {
  // calculo de deltas

  // deltas para la capa de salida
  Capa &capaSalida = capas.back();
  for (int k = 0; k < capaSalida.obtenerNumNeuronas(); ++k) {
    Neurona &neuronaK = capaSalida.neuronas[k];
    if (capaSalida.tipoActivacionCapa == "softmax") {
      neuronaK.delta = neuronaK.salida - salidasEsperadas[k];
    } else {
      neuronaK.delta = (neuronaK.salida - salidasEsperadas[k]) * neuronaK.calcularDerivadaActivacionSalida();
    }
  }

  // deltas para las capas ocultas
  for (int l = capas.size() - 2; l >= 0; --l) {
    Capa &capaActual = capas[l];
    Capa &capaSiguiente = capas[l + 1];
    for (int j = 0; j < capaActual.obtenerNumNeuronas(); ++j) {
      if (capaActual.dropoutRate > 0.0 && !capaActual.dropoutMask[j]) {
        capaActual.neuronas[j].delta = 0.0;
        continue;
      }
      double errorPropagado = 0.0;
      for (int k = 0; k < capaSiguiente.obtenerNumNeuronas(); ++k) {
        errorPropagado += capaSiguiente.neuronas[k].pesos[j] * capaSiguiente.neuronas[k].delta;
      }
      capaActual.neuronas[j].delta = errorPropagado * capaActual.neuronas[j].calcularDerivadaActivacionSalida();
    }
  }

  // acumulacion de gradientes
  for (size_t l = 0; l < capas.size(); ++l) {
    const vector<double> &entradasAEstaCapa = (l == 0) ? entradasMuestra : capas[l - 1].obtenerSalidas();
    for (int j = 0; j < capas[l].obtenerNumNeuronas(); ++j) {
      Neurona &neuronaJ = capas[l].neuronas[j];

      // acumular gradiente del sesgo
      acum_grad_sesgos[l][j] += neuronaJ.delta;

      // acumular gradientes de los pesos
      for (size_t i = 0; i < neuronaJ.pesos.size(); ++i) {
        acum_grad_pesos[l][j][i] += neuronaJ.delta * entradasAEstaCapa[i];
      }
    }
  }
}

void PerceptronMulticapa::aplicar_gradientes_promediados(const vector<vector<vector<double>>> &grad_pesos,
                                                         const vector<vector<double>> &grad_sesgos, int tamano_lote) {
  if (tamano_lote == 0)
    return;

  for (size_t l = 0; l < capas.size(); ++l) {
#pragma omp parallel for
    for (int j = 0; j < capas[l].obtenerNumNeuronas(); ++j) {
      Neurona &neuronaJ = capas[l].neuronas[j];

      // promediar gradientes
      double grad_sesgo_promedio = grad_sesgos[l][j] / tamano_lote;
      vector<double> grad_pesos_promedio(neuronaJ.pesos.size());
      for (size_t i = 0; i < neuronaJ.pesos.size(); ++i) {
        grad_pesos_promedio[i] = grad_pesos[l][j][i] / tamano_lote;
      }

      // aplicar logica del optimizador
      if (tipoOptimizador == "sgd") {
        for (size_t i = 0; i < neuronaJ.pesos.size(); ++i) {
          neuronaJ.pesos[i] -= tasaAprendizaje * grad_pesos_promedio[i];
        }
        neuronaJ.sesgo -= tasaAprendizaje * grad_sesgo_promedio;

      } else if (tipoOptimizador == "rmsprop") {
        const double uno_menos_beta = 1.0 - beta;
        for (size_t i = 0; i < neuronaJ.pesos.size(); ++i) {
          neuronaJ.cachePesos[i] =
              beta * neuronaJ.cachePesos[i] + uno_menos_beta * (grad_pesos_promedio[i] * grad_pesos_promedio[i]);
          neuronaJ.pesos[i] -= (tasaAprendizaje / (sqrt(neuronaJ.cachePesos[i]) + epsilon)) * grad_pesos_promedio[i];
        }
        neuronaJ.cacheSesgo = beta * neuronaJ.cacheSesgo + uno_menos_beta * (grad_sesgo_promedio * grad_sesgo_promedio);
        neuronaJ.sesgo -= (tasaAprendizaje / (sqrt(neuronaJ.cacheSesgo) + epsilon)) * grad_sesgo_promedio;

      } else if (tipoOptimizador == "adam") {
        // añadir Weight Decay (L2)
        for (size_t i = 0; i < neuronaJ.pesos.size(); ++i) {
          grad_pesos_promedio[i] += this->weightDecay * neuronaJ.pesos[i];
        }

        const double bias_correction_factor_m = 1.0 / (1.0 - pow(this->beta1, this->t));
        const double bias_correction_factor_v = 1.0 / (1.0 - pow(this->beta2, this->t));

        // actualizar pesos
        for (size_t i = 0; i < neuronaJ.pesos.size(); ++i) {
          neuronaJ.m_pesos[i] = beta1 * neuronaJ.m_pesos[i] + (1.0 - beta1) * grad_pesos_promedio[i];
          neuronaJ.v_pesos[i] = beta2 * neuronaJ.v_pesos[i] + (1.0 - beta2) * (grad_pesos_promedio[i] * grad_pesos_promedio[i]);
          double m_corr = neuronaJ.m_pesos[i] * bias_correction_factor_m;
          double v_corr = neuronaJ.v_pesos[i] * bias_correction_factor_v;
          neuronaJ.pesos[i] -= (tasaAprendizaje * m_corr) / (sqrt(v_corr) + epsilon);
        }

        // actualizar sesgo
        neuronaJ.m_sesgo = beta1 * neuronaJ.m_sesgo + (1.0 - beta1) * grad_sesgo_promedio;
        neuronaJ.v_sesgo = beta2 * neuronaJ.v_sesgo + (1.0 - beta2) * (grad_sesgo_promedio * grad_sesgo_promedio);
        double m_sesgo_corr = neuronaJ.m_sesgo * bias_correction_factor_m;
        double v_sesgo_corr = neuronaJ.v_sesgo * bias_correction_factor_v;
        neuronaJ.sesgo -= (tasaAprendizaje * m_sesgo_corr) / (sqrt(v_sesgo_corr) + epsilon);
      }
    }
  }
}

vector<double> PerceptronMulticapa::predecir(const vector<double> &entrada) {
  this->enEntrenamiento = false;
  return propagacionAdelante(entrada);
}

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
      // cout << neurona.sesgo << endl;
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
  if (historialPerdida.empty()) {
    cout << "Informacion: El historial de entrenamiento esta vacio. No hay nada que guardar." << endl;
    return;
  }
  // Validación de consistencia de tamaños
  if (historialPerdida.size() != historialPrecision.size() || historialPerdida.size() != historialPerdidaPrueba.size() ||
      historialPerdida.size() != historialPrecisionPrueba.size()) {
    cerr << "Error: Los historiales tienen tamanos inconsistentes. No se guardara el historial." << endl;
    return;
  }

  ofstream archivoSalida(nombreArchivo);
  if (!archivoSalida.is_open()) {
    cerr << "Error: No se pudo abrir el archivo para guardar el historial: " << nombreArchivo << endl;
    return;
  }

  archivoSalida << fixed << setprecision(10);

  // Escribir el encabezado del archivo CSV
  archivoSalida << "Epoca,Perdida_Train,Precision_Train,Perdida_Test,Precision_Test" << endl;

  // Escribir los datos de cada época
  for (size_t i = 0; i < historialPerdida.size(); ++i) {
    archivoSalida << (i + 1) // Numero de epoca
                  << "," << historialPerdida[i] << "," << historialPrecision[i] << "," << historialPerdidaPrueba[i] << ","
                  << historialPrecisionPrueba[i] << endl;
  }

  archivoSalida.close();
  cout << "Historial de entrenamiento completo guardado correctamente en: " << nombreArchivo << endl;
}

pair<double, double> PerceptronMulticapa::evaluar(const vector<vector<double>> &entradasPrueba,
                                                  const vector<vector<double>> &salidasPrueba) {
  bool estadoPrevio = this->enEntrenamiento; // Guarda el estado actual
  this->enEntrenamiento = false;             // Establece el modo de evaluación
  if (entradasPrueba.empty()) {
    return {0.0, 0.0};
  }

  double perdidaTotal = 0.0;
  int prediccionesCorrectas = 0;

  for (size_t i = 0; i < entradasPrueba.size(); ++i) {
    vector<double> salidasActuales = propagacionAdelante(entradasPrueba[i]);

    // Calcular la pérdida Cross-Entropy
    double perdidaMuestra = 0.0;
    for (size_t k = 0; k < salidasActuales.size(); ++k) {
      if (salidasPrueba[i][k] == 1.0) {
        // añadir un pequeño valor (epsilon) para evitar log(0)
        perdidaMuestra -= log(salidasActuales[k] + 1e-9);
      }
    }
    perdidaTotal += perdidaMuestra;

    // Comparar la prediccion con la etiqueta real
    auto itPredicho = max_element(salidasActuales.begin(), salidasActuales.end());
    int digitoPredicho = distance(salidasActuales.begin(), itPredicho);

    auto itEsperado = max_element(salidasPrueba[i].begin(), salidasPrueba[i].end());
    int digitoEsperado = distance(salidasPrueba[i].begin(), itEsperado);

    if (digitoPredicho == digitoEsperado) {
      prediccionesCorrectas++;
    }
  }

  double perdidaPromedio = perdidaTotal / entradasPrueba.size();
  double precision = static_cast<double>(prediccionesCorrectas) / entradasPrueba.size();

  this->enEntrenamiento = estadoPrevio; // restaura el estado original

  return {perdidaPromedio, precision};
}
