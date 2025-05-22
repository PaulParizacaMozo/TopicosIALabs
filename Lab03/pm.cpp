#include "pm.hpp"
#include <algorithm>
#include <cmath> // exp, tanh, max
#include <iostream>
#include <random>
#include <stdexcept>
using namespace std;

// Constructor de la Red Neuronal Multicapa
PerceptronMulticapa::PerceptronMulticapa(int tamano_entrada, int tamano_capa_oculta, int tamano_salida,
                                         double tasa_aprendizaje_inicial, const string &nombre_funcion_activacion,
                                         unsigned int semilla_para_aleatorios)
    : neuronas_entrada_(tamano_entrada), neuronas_capa_oculta_(tamano_capa_oculta), neuronas_salida_(tamano_salida),
      tasa_aprendizaje_(tasa_aprendizaje_inicial) {

  if (tamano_entrada <= 0 || tamano_capa_oculta <= 0 || tamano_salida <= 0) {
    throw invalid_argument("Las dimensiones de las capas deben ser numeros positivos.");
  }
  if (tasa_aprendizaje_inicial <= 0) {
    throw invalid_argument("La tasa de aprendizaje debe ser un numero positivo.");
  }

  configurar_funciones_de_activacion(nombre_funcion_activacion); // Establece las funciones a usar
  inicializar_todos_los_pesos(semilla_para_aleatorios);          // Inicializa pesos y sesgos con la semilla

  entrada_neta_capa_oculta_.resize(neuronas_capa_oculta_);
  activacion_capa_oculta_.resize(neuronas_capa_oculta_);
  entrada_neta_capa_salida_.resize(neuronas_salida_);
  salida_predicha_por_red_.resize(neuronas_salida_);
}

// Configura la funcion de activacion y su derivada segun el nombre especificado
void PerceptronMulticapa::configurar_funciones_de_activacion(const string &nombre_funcion) {
  if (nombre_funcion == "sigmoide") {
    funcion_de_activacion_ = calcular_sigmoide;
    derivada_funcion_activacion_ = calcular_derivada_sigmoide;
  } else if (nombre_funcion == "tanh") {
    funcion_de_activacion_ = calcular_tanh;
    derivada_funcion_activacion_ = calcular_derivada_tanh;
  } else if (nombre_funcion == "relu") {
    funcion_de_activacion_ = calcular_relu;
    derivada_funcion_activacion_ = calcular_derivada_relu;
  } else {
    throw invalid_argument("Nombre de funcion de activacion desconocido: " + nombre_funcion);
  }
}

// Inicializa los pesos y sesgos de la red con valores aleatorios pequeños usando una semilla
void PerceptronMulticapa::inicializar_todos_los_pesos(unsigned int semilla) {
  // Uso de semilla
  mt19937 generador_numeros_aleatorios(semilla);
  uniform_real_distribution<> distribucion_pesos(-0.5, 0.5); // Para pesos
  uniform_real_distribution<> distribucion_sesgos(0.0, 0.1); // Para sesgos

  // Inicializar los pesos de la capa de entrada a capa oculta
  pesos_entrada_a_oculta_.resize(neuronas_capa_oculta_, Vector(neuronas_entrada_));
  for (int i = 0; i < neuronas_capa_oculta_; ++i) {
    for (int j = 0; j < neuronas_entrada_; ++j) {
      pesos_entrada_a_oculta_[i][j] = distribucion_pesos(generador_numeros_aleatorios);
    }
  }

  // Inicializar los sesgos de la capa oculta con valores aleatorios entre 0 y 0.5
  sesgos_capa_oculta_.resize(neuronas_capa_oculta_);
  for (int i = 0; i < neuronas_capa_oculta_; ++i) {
    sesgos_capa_oculta_[i] = distribucion_sesgos(generador_numeros_aleatorios);
  }

  // Inicializar los pesos de la capa oculta a capa de salida
  pesos_oculta_a_salida_.resize(neuronas_salida_, Vector(neuronas_capa_oculta_));
  for (int i = 0; i < neuronas_salida_; ++i) {
    for (int j = 0; j < neuronas_capa_oculta_; ++j) {
      pesos_oculta_a_salida_[i][j] = distribucion_pesos(generador_numeros_aleatorios);
    }
  }

  // Inicializar los sesgos de la capa de salida con valores aleatorios entre 0 y 0.5
  sesgos_capa_salida_.resize(neuronas_salida_);
  for (int i = 0; i < neuronas_salida_; ++i) {
    sesgos_capa_salida_[i] = distribucion_sesgos(generador_numeros_aleatorios);
  }
}

// Ejecuta la propagacion hacia adelante
void PerceptronMulticapa::propagacion_hacia_adelante(const Vector &entrada_actual) {
  if (entrada_actual.size() != neuronas_entrada_) {
    throw invalid_argument("Tamano de vector de entrada incorrecto en propagacion hacia adelante.");
  }

  entrada_neta_capa_oculta_ =
      sumar_dos_vectores(multiplicar_matriz_por_vector(pesos_entrada_a_oculta_, entrada_actual), sesgos_capa_oculta_);
  activacion_capa_oculta_ = aplicar_funcion_a_elementos_vector(entrada_neta_capa_oculta_, funcion_de_activacion_);

  entrada_neta_capa_salida_ =
      sumar_dos_vectores(multiplicar_matriz_por_vector(pesos_oculta_a_salida_, activacion_capa_oculta_), sesgos_capa_salida_);
  salida_predicha_por_red_ = aplicar_funcion_a_elementos_vector(entrada_neta_capa_salida_, funcion_de_activacion_);
}

// Metodo publico para obtener una prediccion
Vector PerceptronMulticapa::predecir(const Vector &entrada_actual) {
  propagacion_hacia_adelante(entrada_actual);
  return salida_predicha_por_red_;
}

// Ejecuta la retropropagacion para ajustar los pesos y sesgos
void PerceptronMulticapa::retropropagacion(const Vector &entrada_actual, const Vector &salida_objetivo_actual) {
  if (salida_objetivo_actual.size() != neuronas_salida_) {
    throw invalid_argument("Tamano del vector de salida objetivo incorrecto en retropropagacion.");
  }
  if (entrada_actual.size() != neuronas_entrada_) {
    throw invalid_argument("Tamano del vector de entrada incorrecto en retropropagacion.");
  }

  Vector error_en_capa_salida = restar_dos_vectores(salida_objetivo_actual, salida_predicha_por_red_);
  Vector derivada_en_salida = aplicar_funcion_a_elementos_vector(entrada_neta_capa_salida_, derivada_funcion_activacion_);
  Vector delta_capa_salida = multiplicar_vectores_elemento_a_elemento(error_en_capa_salida, derivada_en_salida);

  Matriz gradiente_pesos_oculta_salida = calcular_producto_externo_vectores(delta_capa_salida, activacion_capa_oculta_);
  Vector gradiente_sesgos_salida = delta_capa_salida;

  Matriz pesos_oculta_salida_transpuestos = transponer_una_matriz(pesos_oculta_a_salida_);
  Vector error_propagado_a_oculta = multiplicar_matriz_por_vector(pesos_oculta_salida_transpuestos, delta_capa_salida);
  Vector derivada_en_oculta = aplicar_funcion_a_elementos_vector(entrada_neta_capa_oculta_, derivada_funcion_activacion_);
  Vector delta_capa_oculta = multiplicar_vectores_elemento_a_elemento(error_propagado_a_oculta, derivada_en_oculta);

  Matriz gradiente_pesos_entrada_oculta = calcular_producto_externo_vectores(delta_capa_oculta, entrada_actual);
  Vector gradiente_sesgos_oculta = delta_capa_oculta;

  pesos_oculta_a_salida_ = sumar_dos_matrices(pesos_oculta_a_salida_,
                                              multiplicar_escalar_por_matriz(tasa_aprendizaje_, gradiente_pesos_oculta_salida));
  sesgos_capa_salida_ =
      sumar_dos_vectores(sesgos_capa_salida_, multiplicar_escalar_por_vector(tasa_aprendizaje_, gradiente_sesgos_salida));

  pesos_entrada_a_oculta_ = sumar_dos_matrices(
      pesos_entrada_a_oculta_, multiplicar_escalar_por_matriz(tasa_aprendizaje_, gradiente_pesos_entrada_oculta));
  sesgos_capa_oculta_ =
      sumar_dos_vectores(sesgos_capa_oculta_, multiplicar_escalar_por_vector(tasa_aprendizaje_, gradiente_sesgos_oculta));
}

// Metodo principal para entrenar la red neuronal
void PerceptronMulticapa::entrenar(const vector<Vector> &conjunto_entradas_entrenamiento,
                                   const vector<Vector> &conjunto_salidas_objetivo, int numero_epocas) {
  if (conjunto_entradas_entrenamiento.size() != conjunto_salidas_objetivo.size()) {
    throw invalid_argument("El numero de muestras de entrada y salida para entrenamiento debe ser el mismo.");
  }
  if (conjunto_entradas_entrenamiento.empty()) {
    cout << "Advertencia: No se proporcionaron datos de entrenamiento." << endl;
    return;
  }
  imprimir_pesos_actuales(); // pesos iniciales antes de entrenar
  for (int epoca_actual = 0; epoca_actual < numero_epocas; ++epoca_actual) {
    double error_total_epoca = 0.0;
    for (size_t i = 0; i < conjunto_entradas_entrenamiento.size(); ++i) {
      propagacion_hacia_adelante(conjunto_entradas_entrenamiento[i]);
      for (size_t j = 0; j < neuronas_salida_; ++j) {
        error_total_epoca += 0.5 * pow(conjunto_salidas_objetivo[i][j] - salida_predicha_por_red_[j], 2);
      }
      retropropagacion(conjunto_entradas_entrenamiento[i], conjunto_salidas_objetivo[i]);
    }
    if ((epoca_actual + 1) % 1000 == 0 || epoca_actual == 0) {
      cout << "Epoca " << epoca_actual + 1 << "/" << numero_epocas
           << ", Error Promedio Muestras: " << error_total_epoca / conjunto_entradas_entrenamiento.size() << endl;
    }
  }

  imprimir_pesos_actuales(); // pesos finales
}

// Imprime los pesos y sesgos actuales de la red
void PerceptronMulticapa::imprimir_pesos_actuales() const {
  cout << "Pesos Conexiones Entrada -> Capa Oculta (pesos_entrada_a_oculta_):\n";
  for (const auto &fila_pesos : pesos_entrada_a_oculta_) {
    for (double valor_peso : fila_pesos) {
      cout << valor_peso << "\t";
    }
    cout << "\n";
  }
  cout << "Sesgos Neuronas Capa Oculta (sesgos_capa_oculta_):\n";
  for (double valor_sesgo : sesgos_capa_oculta_) {
    cout << valor_sesgo << "\t";
  }
  cout << "\n\nPesos Conexiones Capa Oculta -> Capa Salida (pesos_oculta_a_salida_):\n";
  for (const auto &fila_pesos : pesos_oculta_a_salida_) {
    for (double valor_peso : fila_pesos) {
      cout << valor_peso << "\t";
    }
    cout << "\n";
  }
  cout << "Sesgos Neuronas Capa Salida (sesgos_capa_salida_):\n";
  for (double valor_sesgo : sesgos_capa_salida_) {
    cout << valor_sesgo << "\t";
  }
  cout << "\n" << endl;
}

// --- Implementacion de las funciones de activacion y sus derivadas (metodos estaticos) ---
double PerceptronMulticapa::calcular_sigmoide(double valor_x) { return 1.0 / (1.0 + exp(-valor_x)); }
double PerceptronMulticapa::calcular_derivada_sigmoide(double valor_x) {
  double sig_x = calcular_sigmoide(valor_x);
  return sig_x * (1.0 - sig_x);
}
double PerceptronMulticapa::calcular_tanh(double valor_x) { return tanh(valor_x); }
double PerceptronMulticapa::calcular_derivada_tanh(double valor_x) {
  double tanh_x = tanh(valor_x);
  return 1.0 - tanh_x * tanh_x;
}
double PerceptronMulticapa::calcular_relu(double valor_x) { return max(0.0, valor_x); }
double PerceptronMulticapa::calcular_derivada_relu(double valor_x) { return (valor_x > 0.0) ? 1.0 : 0.0; }

// --- Implementacion de las funciones de operaciones con matrices ---
Vector PerceptronMulticapa::aplicar_funcion_a_elementos_vector(const Vector &vector_actual,
                                                               const function<double(double)> &funcion_a_aplicar) {
  Vector vector_resultado(vector_actual.size());
  transform(vector_actual.begin(), vector_actual.end(), vector_resultado.begin(), funcion_a_aplicar);
  return vector_resultado;
}

// Operaciones con matrices
Vector PerceptronMulticapa::multiplicar_matriz_por_vector(const Matriz &matriz_actual, const Vector &vector_actual) {
  if (matriz_actual.empty() || matriz_actual[0].size() != vector_actual.size()) {
    throw runtime_error("Dimensiones incompatibles para multiplicacion matriz-vector.");
  }
  size_t numero_filas = matriz_actual.size();
  size_t numero_columnas = matriz_actual[0].size();
  Vector vector_resultado(numero_filas, 0.0);
  for (size_t i = 0; i < numero_filas; ++i) {
    for (size_t j = 0; j < numero_columnas; ++j) {
      vector_resultado[i] += matriz_actual[i][j] * vector_actual[j];
    }
  }
  return vector_resultado;
}

Vector PerceptronMulticapa::sumar_dos_vectores(const Vector &vector_uno, const Vector &vector_dos) {
  if (vector_uno.size() != vector_dos.size()) {
    throw runtime_error("Dimensiones incompatibles para suma de vectores.");
  }
  Vector vector_resultado(vector_uno.size());
  for (size_t i = 0; i < vector_uno.size(); ++i) {
    vector_resultado[i] = vector_uno[i] + vector_dos[i];
  }
  return vector_resultado;
}

Vector PerceptronMulticapa::restar_dos_vectores(const Vector &vector_uno, const Vector &vector_dos) {
  if (vector_uno.size() != vector_dos.size()) {
    throw runtime_error("Dimensiones incompatibles para resta de vectores.");
  }
  Vector vector_resultado(vector_uno.size());
  for (size_t i = 0; i < vector_uno.size(); ++i) {
    vector_resultado[i] = vector_uno[i] - vector_dos[i];
  }
  return vector_resultado;
}

Vector PerceptronMulticapa::multiplicar_vectores_elemento_a_elemento(const Vector &vector_uno, const Vector &vector_dos) {
  if (vector_uno.size() != vector_dos.size()) {
    throw runtime_error("Dimensiones incompatibles para multiplicacion elemento a elemento de vectores.");
  }
  Vector vector_resultado(vector_uno.size());
  for (size_t i = 0; i < vector_uno.size(); ++i) {
    vector_resultado[i] = vector_uno[i] * vector_dos[i];
  }
  return vector_resultado;
}

Vector PerceptronMulticapa::multiplicar_escalar_por_vector(double valor_escalar, const Vector &vector_actual) {
  Vector vector_resultado = vector_actual;
  for (double &valor_elemento : vector_resultado) {
    valor_elemento *= valor_escalar;
  }
  return vector_resultado;
}

Matriz PerceptronMulticapa::multiplicar_escalar_por_matriz(double valor_escalar, const Matriz &matriz_actual) {
  Matriz matriz_resultado = matriz_actual;
  for (auto &fila_matriz : matriz_resultado) {
    for (double &valor_elemento : fila_matriz) {
      valor_elemento *= valor_escalar;
    }
  }
  return matriz_resultado;
}

Matriz PerceptronMulticapa::calcular_producto_externo_vectores(const Vector &vector_columna,
                                                               const Vector &vector_fila_como_columna) {
  Matriz matriz_resultado(vector_columna.size(), Vector(vector_fila_como_columna.size()));
  for (size_t i = 0; i < vector_columna.size(); ++i) {
    for (size_t j = 0; j < vector_fila_como_columna.size(); ++j) {
      matriz_resultado[i][j] = vector_columna[i] * vector_fila_como_columna[j];
    }
  }
  return matriz_resultado;
}

Matriz PerceptronMulticapa::transponer_una_matriz(const Matriz &matriz_original) {
  if (matriz_original.empty())
    return Matriz();
  size_t numero_filas = matriz_original.size();
  size_t numero_columnas = matriz_original[0].size();
  Matriz matriz_resultado(numero_columnas, Vector(numero_filas));
  for (size_t i = 0; i < numero_filas; ++i) {
    for (size_t j = 0; j < numero_columnas; ++j) {
      matriz_resultado[j][i] = matriz_original[i][j];
    }
  }
  return matriz_resultado;
}

Matriz PerceptronMulticapa::sumar_dos_matrices(const Matriz &matriz_una, const Matriz &matriz_dos) {
  if (matriz_una.size() != matriz_dos.size() || (!matriz_una.empty() && matriz_una[0].size() != matriz_dos[0].size())) {
    throw runtime_error("Dimensiones incompatibles para suma de matrices.");
  }
  Matriz matriz_resultado = matriz_una;
  for (size_t i = 0; i < matriz_una.size(); ++i) {
    for (size_t j = 0; j < matriz_una[0].size(); ++j) {
      matriz_resultado[i][j] += matriz_dos[i][j];
    }
  }
  return matriz_resultado;
}

Matriz PerceptronMulticapa::restar_dos_matrices(const Matriz &matriz_una, const Matriz &matriz_dos) {
  if (matriz_una.size() != matriz_dos.size() || (!matriz_una.empty() && matriz_una[0].size() != matriz_dos[0].size())) {
    throw runtime_error("Dimensiones incompatibles para resta de matrices.");
  }
  Matriz matriz_resultado = matriz_una;
  for (size_t i = 0; i < matriz_una.size(); ++i) {
    for (size_t j = 0; j < matriz_una[0].size(); ++j) {
      matriz_resultado[i][j] -= matriz_dos[i][j];
    }
  }
  return matriz_resultado;
}
