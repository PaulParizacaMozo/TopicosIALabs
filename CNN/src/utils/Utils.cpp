#include "core/Tensor.hpp"
#include "losses/CrossEntropy.hpp" // función softmax
#include "model/Sequential.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Guarda los parámetros (pesos y biases) de un modelo en un archivo binario.
 * @details El formato del archivo es:
 *          1. [size_t] Número total de tensores de parámetros.
 *          2. Para cada tensor:
 *             a. [size_t] Rango del tensor (número de dimensiones).
 *             b. [size_t*] La forma del tensor.
 *             c. [float*] Los datos del tensor.
 * @param model El modelo a guardar.
 * @param filePath La ruta del archivo donde se guardará el modelo.
 */
void saveModel(const Sequential &model, const std::string &filePath) {
  std::ofstream outFile(filePath, std::ios::binary);
  if (!outFile) {
    throw std::runtime_error("Error: No se pudo abrir el archivo para escritura: " + filePath);
  }
  std::cout << "Guardando modelo en: " << filePath << std::endl;

  // Se necesita una referencia no-const para llamar a getParameters.
  // NOTA: Una mejora a futuro sería hacer `getParameters()` const.
  Sequential &nonConstModel = const_cast<Sequential &>(model);
  std::vector<Tensor *> parameters = nonConstModel.getParameters();

  // Escribir el número de tensores
  size_t numTensors = parameters.size();
  outFile.write(reinterpret_cast<const char *>(&numTensors), sizeof(size_t));

  // Escribir cada tensor (rango, forma, datos)
  for (const auto &param : parameters) {
    size_t rank = param->getShape().size();
    outFile.write(reinterpret_cast<const char *>(&rank), sizeof(size_t));
    outFile.write(reinterpret_cast<const char *>(param->getShape().data()), rank * sizeof(size_t));

    size_t numElements = param->getSize();
    outFile.write(reinterpret_cast<const char *>(param->getData()), numElements * sizeof(float));
  }

  outFile.close();
  std::cout << "Modelo guardado con éxito." << std::endl;
}

/**
 * @brief Carga los parámetros en un modelo desde un archivo binario.
 * @warning La arquitectura del `model` que se pasa ya debe ser idéntica a la del
 *          modelo guardado. Esta función solo carga los valores de los pesos.
 * @param model El modelo (ya construido) en el que se cargarán los pesos.
 * @param filePath La ruta del archivo desde donde se cargará el modelo.
 */
void loadModel(Sequential &model, const std::string &filePath) {
  std::ifstream inFile(filePath, std::ios::binary);
  if (!inFile) {
    throw std::runtime_error("Error: No se pudo abrir el archivo para lectura: " + filePath);
  }
  std::cout << "Cargando modelo desde: " << filePath << std::endl;

  std::vector<Tensor *> parameters = model.getParameters();

  // Verificar que el número de tensores coincida
  size_t numTensorsInFile;
  inFile.read(reinterpret_cast<char *>(&numTensorsInFile), sizeof(size_t));
  if (numTensorsInFile != parameters.size()) {
    throw std::runtime_error("Error: La arquitectura del modelo no coincide con el archivo.");
  }

  // Leer cada tensor y cargarlo en el modelo
  for (auto &param : parameters) {
    size_t rank;
    inFile.read(reinterpret_cast<char *>(&rank), sizeof(size_t));

    std::vector<size_t> shapeInFile(rank);
    inFile.read(reinterpret_cast<char *>(shapeInFile.data()), rank * sizeof(size_t));

    // Verificar que la forma de cada tensor coincida
    if (shapeInFile != param->getShape()) {
      throw std::runtime_error("Error: La forma de un parámetro en el archivo no coincide.");
    }

    // Leer los datos directamente en la memoria del tensor del modelo.
    size_t numElements = param->getSize();
    inFile.read(reinterpret_cast<char *>(param->getData()), numElements * sizeof(float));
  }

  inFile.close();
  std::cout << "Modelo cargado con éxito." << std::endl;
}

/**
 * @brief Dibuja una imagen de MNIST/Fashion-MNIST en la consola y predice su clase.
 * @details Esta función es una herramienta de depuración y visualización que soporta
 *          tanto entradas 2D (para MLPs) como 4D (para CNNs).
 * @param model El modelo entrenado a usar para la predicción.
 * @param X_test El conjunto de datos de prueba.
 * @param y_test Las etiquetas de prueba.
 * @param index El índice de la muestra a visualizar y predecir.
 */
void predictAndDraw(Sequential &model, const Tensor &X_test, const Tensor &y_test, size_t index) {
  if (index >= X_test.getShape()[0]) {
    throw std::out_of_range("Índice de muestra fuera de rango.");
  }

  std::cout << "\n--- Visualizando y Prediciendo Muestra #" << index << " ---\n" << std::endl;

  // 1. Dibujar la imagen en la consola
  Tensor sample = X_test.slice(index, 1);
  const auto &sampleShape = sample.getShape();
  const char grayscale_ramp[] = " .:-=+*#%@"; // Rampa de caracteres para simular escala de grises.

  std::cout << "Imagen de Entrada (28x28):" << std::endl;
  if (sampleShape.size() == 2 && sampleShape[1] == 784) { // Lógica para entrada 2D aplanada
    for (size_t h = 0; h < 28; ++h) {
      for (size_t w = 0; w < 28; ++w) {
        float pixel = sample(0, h * 28 + w);
        int ramp_idx = static_cast<int>(pixel * (sizeof(grayscale_ramp) - 2));
        std::cout << grayscale_ramp[ramp_idx] << grayscale_ramp[ramp_idx];
      }
      std::cout << std::endl;
    }
  } else if (sampleShape.size() == 4 && sampleShape[2] == 28 && sampleShape[3] == 28) { // Lógica para entrada 4D
    for (size_t h = 0; h < 28; ++h) {
      for (size_t w = 0; w < 28; ++w) {
        // Se visualiza solo el primer canal si hay múltiples.
        float pixel = sample(0, 0, h, w);
        int ramp_idx = static_cast<int>(pixel * (sizeof(grayscale_ramp) - 2));
        std::cout << grayscale_ramp[ramp_idx] << grayscale_ramp[ramp_idx];
      }
      std::cout << std::endl;
    }
  } else {
    std::cout << "  [No se puede dibujar. Forma de tensor no reconocida: " << sample.shapeToString() << "]" << std::endl;
  }

  // 2. Realizar predicción
  Tensor prediction_logits = model.predict(sample);
  Tensor probabilities = softmax(prediction_logits);

  // 3. Obtener clase predicha y real
  auto argmax_func = [](const float *data, size_t size) { return std::distance(data, std::max_element(data, data + size)); };
  size_t predicted_class = argmax_func(probabilities.getData(), probabilities.getShape()[1]);
  size_t true_class = argmax_func(y_test.getData() + index * y_test.getShape()[1], y_test.getShape()[1]);

  // 4. Mostrar resultados
  std::cout << "\n----------------------------------------" << std::endl;
  std::cout << "Etiqueta Real:         " << true_class << std::endl;
  std::cout << "Predicción del Modelo: " << predicted_class << " ("
            << (predicted_class == true_class ? "CORRECTO" : "INCORRECTO") << ")" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Probabilidades de salida:" << std::endl;
  for (size_t i = 0; i < probabilities.getShape()[1]; ++i) {
    std::cout << "  Clase " << i << ": " << std::fixed << std::setprecision(2) << probabilities(0, i) * 100.0f << "%"
              << std::endl;
  }
  std::cout << "========================================\n" << std::endl;
}
