#ifndef PM_HPP
#define PM_HPP

#include <functional> // para function
#include <string>
#include <vector>
using namespace std;

// Alias para los tipos de datos que usaremos
using Vector = vector<double>;
using Matriz = vector<vector<double>>;

class PerceptronMulticapa {
public:
  // Constructor incluye nombre de funcion de activacion y semilla
  PerceptronMulticapa(int tamano_entrada, int tamano_capa_oculta, int tamano_salida, double tasa_aprendizaje_inicial,
                      const string &nombre_funcion_activacion, // "sigmoide", "tanh", "relu"
                      unsigned int semilla_para_aleatorios);   // Semilla para reproducibilidad

  // Metodo para predecir la salida dada una entrada
  Vector predecir(const Vector &entrada_actual);

  // Metodo para entrenar la red
  void entrenar(const vector<Vector> &conjunto_entradas_entrenamiento, const vector<Vector> &conjunto_salidas_objetivo,
                int numero_epocas);

  // Metodo para mostrar los pesos actuales de la red (util para depuracion)
  void imprimir_pesos_actuales() const;

private:
  // --- Parametros de la arquitectura de la red ---
  int neuronas_entrada_;
  int neuronas_capa_oculta_;
  int neuronas_salida_;
  double tasa_aprendizaje_;

  // --- Pesos y sesgos (biases) de la red ---
  Matriz pesos_entrada_a_oculta_;
  Vector sesgos_capa_oculta_;
  Matriz pesos_oculta_a_salida_;
  Vector sesgos_capa_salida_;

  // --- Funciones de activacion y sus derivadas (almacenadas como function) ---
  function<double(double)> funcion_de_activacion_;
  function<double(double)> derivada_funcion_activacion_;

  // --- Valores intermedios (para propagacion y retropropagacion) ---
  Vector entrada_neta_capa_oculta_;
  Vector activacion_capa_oculta_;
  Vector entrada_neta_capa_salida_;
  Vector salida_predicha_por_red_;

  // --- Metodos privados internos ---
  void inicializar_todos_los_pesos(unsigned int semilla); // con semilla
  void configurar_funciones_de_activacion(const string &nombre_funcion);

  void propagacion_hacia_adelante(const Vector &entrada_actual);
  void retropropagacion(const Vector &entrada_actual, const Vector &salida_objetivo_actual);

  // --- Funciones de activacion estaticas y sus derivadas ---
  static double calcular_sigmoide(double valor_x);
  static double calcular_derivada_sigmoide(double valor_x);
  static double calcular_tanh(double valor_x);
  static double calcular_derivada_tanh(double valor_x);
  static double calcular_relu(double valor_x);
  static double calcular_derivada_relu(double valor_x);

  // --- Funciones para calculo con matrices ---
  // aplicar una funcion (como sigmoide, tanh, relu) a cada elemento de un vector
  static Vector aplicar_funcion_a_elementos_vector(const Vector &vector_actual,
                                                   const function<double(double)> &funcion_a_aplicar);
  // operaciones con matrices
  static Vector multiplicar_matriz_por_vector(const Matriz &matriz_actual, const Vector &vector_actual);
  static Vector sumar_dos_vectores(const Vector &vector_uno, const Vector &vector_dos);
  static Vector restar_dos_vectores(const Vector &vector_uno, const Vector &vector_dos);
  static Vector multiplicar_vectores_elemento_a_elemento(const Vector &vector_uno, const Vector &vector_dos);
  static Vector multiplicar_escalar_por_vector(double valor_escalar, const Vector &vector_actual);
  static Matriz multiplicar_escalar_por_matriz(double valor_escalar, const Matriz &matriz_actual);
  static Matriz calcular_producto_externo_vectores(const Vector &vector_columna, const Vector &vector_fila_como_columna);
  static Matriz transponer_una_matriz(const Matriz &matriz_original);
  static Matriz sumar_dos_matrices(const Matriz &matriz_una, const Matriz &matriz_dos);
  static Matriz restar_dos_matrices(const Matriz &matriz_una, const Matriz &matriz_dos);
};

#endif // PM_HPP
