#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

// Kernel para suma de vectores
__global__ void sumVectors(const float *ptr_v1, const float *ptr_v2,
                           float *ptr_res, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    ptr_res[index] = ptr_v1[index] + ptr_v2[index];
  }
}

int main() {
  const int N = 1000000; // N elementos

  // Inicializar generador de numeros aleatorios con semilla fija
  std::mt19937 gen(322); // Semilla fija
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // Crear vectores en el host
  std::vector<float> v1(N), v2(N), v_res(N);
  for (int i = 0; i < N; ++i) {
    v1[i] = dist(gen);
    v2[i] = dist(gen);
  }

  // Punteros para memoria en el device (GPU)
  float *ptr_v1, *ptr_v2, *ptr_res;

  // Asignar memoria en la GPU
  cudaMalloc(&ptr_v1, N * sizeof(float));
  cudaMalloc(&ptr_v2, N * sizeof(float));
  cudaMalloc(&ptr_res, N * sizeof(float));

  // Copiar datos desde el host al device
  cudaMemcpy(ptr_v1, v1.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_v2, v2.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  // Configurar grid y block
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // Medir tiempo de ejecucion del kernel
  auto start = std::chrono::high_resolution_clock::now();
  sumVectors<<<gridSize, blockSize>>>(ptr_v1, ptr_v2, ptr_res, N);
  cudaDeviceSynchronize(); // Esperar a que el kernel termine
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Copiar resultado de la GPU al host
  cudaMemcpy(v_res.data(), ptr_res, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Imprimir tiempo de ejecucion
  std::cout << "Tiempo de ejecucion en GPU: " << duration.count()
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

  // Liberar memoria del device
  cudaFree(ptr_v1);
  cudaFree(ptr_v2);
  cudaFree(ptr_res);

  return 0;
}
