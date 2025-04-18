#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// Funcion para sumar vectores en CPU
void sumVectorsCPU(const std::vector<float> &v1, const std::vector<float> &v2,
                   std::vector<float> &v_res, int n) {
  for (int i = 0; i < n; ++i) {
    v_res[i] = v1[i] + v2[i];
  }
}

int main() {
  const int N = 1000000; // N elementos

  // Inicializar generador de numeros aleatorios con semilla fija
  std::mt19937 gen(322); // Semilla
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // Crear vectores
  std::vector<float> v1(N), v2(N), v_res(N);
  for (int i = 0; i < N; ++i) {
    v1[i] = dist(gen);
    v2[i] = dist(gen);
  }

  // Medir tiempo de ejecucion
  auto start = std::chrono::high_resolution_clock::now();
  sumVectorsCPU(v1, v2, v_res, N);
  auto end = std::chrono::high_resolution_clock::now();

  // Calcular duracion
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Tiempo de ejecucion en CPU: " << duration.count()
            << " microsegundos\n";

  // Verificar algunos resultados
  std::cout << "Primeros 3 resultados:\n";
  for (int i = 0; i < 3 && i < N; ++i) {
    std::cout << v1[i] << " + " << v2[i] << " = " << v_res[i] << "\n";
  }

  std::cout << "Ultimos 3 resultados:\n";
  for (int i = std::max(0, N - 3); i < N; ++i) {
    std::cout << v1[i] << " + " << v2[i] << " = " << v_res[i] << "\n";
  }

  return 0;
}
