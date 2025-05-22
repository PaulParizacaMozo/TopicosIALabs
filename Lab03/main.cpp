#include "pm.hpp"
#include <iostream>
#include <string>
#include <vector>
using namespace std;

// Funcion para probar una compuerta logica especifica
void probar_compuerta_logica(const string &nombre_compuerta, const vector<Vector> &conjunto_entradas,
                             const vector<Vector> &conjunto_salidas_esperadas,
                             const string &nombre_funcion_activacion_usar, // string para la funcion
                             double tasa_aprendizaje_para_prueba, int numero_epocas_entrenamiento,
                             unsigned int semilla_usar) { // semilla para la prueba

  cout << "\n--- Probando Compuerta Logica: " << nombre_compuerta << " (Activacion: " << nombre_funcion_activacion_usar
       << ", Semilla: " << semilla_usar << ") ---" << endl;

  // Arquitectura: 2 neuronas de entrada, 2 en capa oculta, 1 de salida
  PerceptronMulticapa red_neuronal(2, 2, 1, tasa_aprendizaje_para_prueba, nombre_funcion_activacion_usar, semilla_usar);

  cout << "Iniciando entrenamiento de la red..." << endl;
  red_neuronal.entrenar(conjunto_entradas, conjunto_salidas_esperadas, numero_epocas_entrenamiento);

  cout << "\nResultados de Prediccion para la Compuerta " << nombre_compuerta << ":" << endl;
  cout << "Entrada\t\tSalida Esperada\t\tPrediccion de la Red" << endl;
  cout << "------------------------------------------------------------" << endl;
  // cout << fixed << setprecision(5);

  for (size_t i = 0; i < conjunto_entradas.size(); ++i) {
    Vector prediccion_actual = red_neuronal.predecir(conjunto_entradas[i]);
    cout << "[" << conjunto_entradas[i][0] << ", " << conjunto_entradas[i][1] << "]\t\t" << conjunto_salidas_esperadas[i][0]
         << "\t\t\t" << prediccion_actual[0] << endl;
  }
  cout << "------------------------------------------------------------" << endl;
}

int main() {
  // --- Configuracion General para las Pruebas ---
  const double TASA_APRENDIZAJE_GLOBAL = 0.1;
  const int NUMERO_EPOCAS_GLOBAL = 15000;
  const unsigned int SEMILLA_FIJA = 1022; // Semilla para resultados reproducibles

  // --- Definicion de los Conjuntos de Datos ---
  vector<Vector> conjunto_entradas_logicas = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  vector<Vector> salidas_esperadas_xor = {{0.0}, {1.0}, {1.0}, {0.0}};
  vector<Vector> salidas_esperadas_and = {{0.0}, {0.0}, {0.0}, {1.0}};
  vector<Vector> salidas_esperadas_or = {{0.0}, {1.0}, {1.0}, {1.0}};

  // Nombres de las funciones de activacion a probar
  // vector<string> nombres_funciones_a_probar = {"relu"};
  vector<string> nombres_funciones_a_probar = {"sigmoide", "tanh", "relu"};

  cout << "***************************************" << endl;
  cout << "** PRUEBAS DE RED NEURONAL MULTICAPA **" << endl;
  cout << "***************************************" << endl;

  // --- Probar Compuerta XOR ---
  cout << "\n\n================== COMPUERTA XOR ==================" << endl;
  for (const string &nombre_func : nombres_funciones_a_probar) {
    probar_compuerta_logica("XOR", conjunto_entradas_logicas, salidas_esperadas_xor, nombre_func, TASA_APRENDIZAJE_GLOBAL,
                            NUMERO_EPOCAS_GLOBAL, SEMILLA_FIJA);
  }
  //// --- Probar Compuerta AND ---
  // cout << "\n\n================== COMPUERTA AND ==================" << endl;
  // for (const string &nombre_func : nombres_funciones_a_probar) {
  //   probar_compuerta_logica("AND", conjunto_entradas_logicas, salidas_esperadas_and, nombre_func, TASA_APRENDIZAJE_GLOBAL,
  //                           NUMERO_EPOCAS_GLOBAL, SEMILLA_FIJA);
  // }

  //// --- Probar Compuerta OR ---
  // cout << "\n\n================== COMPUERTA OR ===================" << endl;
  // for (const string &nombre_func : nombres_funciones_a_probar) {
  //   probar_compuerta_logica("OR", conjunto_entradas_logicas, salidas_esperadas_or, nombre_func, TASA_APRENDIZAJE_GLOBAL,
  //                           NUMERO_EPOCAS_GLOBAL, SEMILLA_FIJA);
  // }

  cout << "\n\n--- Fin de todas las pruebas ---" << endl;

  return 0;
}
