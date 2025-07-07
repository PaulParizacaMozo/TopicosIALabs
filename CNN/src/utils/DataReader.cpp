#include "core/Tensor.hpp"

#include <algorithm> // Para std::random_shuffle
#include <cstdlib>   // Para srand
#include <ctime>     // Para time
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Convierte un vector de etiquetas de clase (enteros) a un formato one-hot.
 * @param labels Vector de etiquetas, donde cada elemento es un entero de clase.
 * @param numClasses El número total de clases posibles.
 * @return Un tensor 2D de forma {num_samples, num_classes} en formato one-hot.
 */
Tensor oneHotEncode(const std::vector<int> &labels, int numClasses) {
  const size_t numSamples = labels.size();
  std::vector<float> oneHotData(numSamples * numClasses, 0.0f);

  for (size_t i = 0; i < numSamples; ++i) {
    if (labels[i] >= 0 && labels[i] < numClasses) {
      // Pone a 1.0f la columna correspondiente a la clase.
      oneHotData[i * numClasses + labels[i]] = 1.0f;
    }
  }
  return Tensor({numSamples, static_cast<size_t>(numClasses)}, oneHotData);
}

/**
 * @brief Carga y procesa un dataset tipo MNIST desde un archivo CSV.
 * @details Lee un CSV donde la primera columna es la etiqueta y las siguientes 784
 *          son los píxeles. Normaliza los píxeles a [0, 1] y codifica las
 *          etiquetas en formato one-hot.
 *
 * @param filePath La ruta al archivo .csv.
 * @param sampleFraction La fracción de los datos a cargar (de 0.0 a 1.0).
 * @param channels El número de canales de salida para las imágenes.
 *        - `channels = 1` (defecto): Genera imágenes en escala de grises {N, 1, 28, 28}.
 *        - `channels = 3`: Genera imágenes "RGB" repitiendo el canal de gris {N, 3, 28, 28}.
 * @param shuffle Si es `true`, baraja los datos antes de tomar la fracción de muestra.
 * @return Un par de Tensores {X, y}, donde X son las imágenes e y las etiquetas.
 */
std::pair<Tensor, Tensor> loadMnist(const std::string &filePath, float sampleFraction = 1.0f, int channels = 1,
                                    bool shuffle = true) {
  if (channels != 1 && channels != 3) {
    throw std::invalid_argument("El número de canales debe ser 1 o 3.");
  }

  std::cout << "Cargando MNIST desde: " << filePath << " (fracción: " << sampleFraction * 100 << "%, canales: " << channels
            << ")" << std::endl;

  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Error: No se pudo abrir el archivo: " + filePath);
  }

  // 1. Lectura del archivo CSV
  std::string line;
  std::getline(file, line); // Ignorar la línea de cabecera

  std::vector<std::vector<float>> allPixelData;
  std::vector<int> allLabels;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string valueStr;

    // Leer la etiqueta
    std::getline(ss, valueStr, ',');
    allLabels.push_back(std::stoi(valueStr));

    // Leer los 784 píxeles y normalizarlos
    std::vector<float> pixels;
    pixels.reserve(784);
    while (std::getline(ss, valueStr, ',')) {
      pixels.push_back(std::stof(valueStr) / 255.0f);
    }
    allPixelData.push_back(pixels);
  }
  file.close();

  // 2. Muestreo aleatorio si se solicita
  size_t totalSamples = allLabels.size();
  if (shuffle && sampleFraction < 1.0f) {
    std::vector<size_t> indices(totalSamples);
    for (size_t i = 0; i < totalSamples; ++i)
      indices[i] = i;

    // Usar un generador de C++11 para un mejor barajado
    std::srand(static_cast<unsigned int>(std::time(0)));
    std::random_shuffle(indices.begin(), indices.end());

    size_t samplesToLoad = static_cast<size_t>(totalSamples * sampleFraction);
    std::vector<std::vector<float>> sampledPixelData;
    std::vector<int> sampledLabels;
    sampledPixelData.reserve(samplesToLoad);
    sampledLabels.reserve(samplesToLoad);

    for (size_t i = 0; i < samplesToLoad; ++i) {
      sampledPixelData.push_back(allPixelData[indices[i]]);
      sampledLabels.push_back(allLabels[indices[i]]);
    }
    allPixelData = std::move(sampledPixelData);
    allLabels = std::move(sampledLabels);
  } else if (sampleFraction < 1.0f) {
    size_t samplesToLoad = static_cast<size_t>(totalSamples * sampleFraction);
    allPixelData.resize(samplesToLoad);
    allLabels.resize(samplesToLoad);
  }

  size_t finalSamples = allLabels.size();

  // 3. Aplanar los datos y manejar los canales
  std::vector<float> flatPixelData;
  flatPixelData.reserve(finalSamples * 784 * channels);
  for (const auto &pixels : allPixelData) {
    for (int c = 0; c < channels; ++c) {
      flatPixelData.insert(flatPixelData.end(), pixels.begin(), pixels.end());
    }
  }

  // 4. Crear los tensores finales
  // La forma del tensor de entrada depende del número de canales solicitados.
  Tensor X({finalSamples, static_cast<size_t>(channels), 28, 28}, flatPixelData);
  Tensor y = oneHotEncode(allLabels, 10);

  std::cout << "Carga completa. " << finalSamples << " muestras cargadas." << std::endl;
  std::cout << "Forma de X: " << X.shapeToString() << ", Forma de y: " << y.shapeToString() << std::endl;

  return {std::move(X), std::move(y)};
}
