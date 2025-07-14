#include "utils/DataReader.hpp"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

// --- Funciones Auxiliares (privadas a este archivo) ---
namespace {
// Convierte un vector de etiquetas de clase (enteros) a un formato one-hot.
Tensor oneHotEncode(const std::vector<int> &labels, int num_classes) {
  const size_t num_samples = labels.size();
  std::vector<float> one_hot_data(num_samples * num_classes, 0.0f);

  for (size_t i = 0; i < num_samples; ++i) {
    if (labels[i] >= 0 && labels[i] < num_classes) {
      one_hot_data[i * num_classes + labels[i]] = 1.0f;
    }
  }
  return Tensor({num_samples, static_cast<size_t>(num_classes)}, one_hot_data);
}
} // namespace

// --- Implementacion de la Funcion Principal ---

std::pair<Tensor, Tensor> load_csv_data(const std::string &filePath, float sample_fraction) {
  std::cout << "Cargando datos desde: " << filePath << " (fraccion a cargar: " << sample_fraction * 100 << "%)" << std::endl;

  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Error: No se pudo abrir el archivo: " + filePath);
  }

  // 1. Leer todas las lineas del archivo CSV en memoria.
  std::string line;
  std::getline(file, line); // Ignorar la linea de cabecera.

  std::vector<std::vector<float>> all_pixel_data;
  std::vector<int> all_labels;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value_str;

    // Leer la etiqueta (primera columna).
    std::getline(ss, value_str, ',');
    all_labels.push_back(std::stoi(value_str));

    // Leer los 784 pixeles y normalizarlos a [0, 1].
    std::vector<float> pixels;
    pixels.reserve(784);
    while (std::getline(ss, value_str, ',')) {
      pixels.push_back(std::stof(value_str) / 255.0f);
    }
    if (pixels.size() != 784) {
      std::cerr << "Advertencia: Fila con numero de pixeles incorrecto. Se ignora." << std::endl;
      all_labels.pop_back(); // Eliminar la etiqueta correspondiente.
      continue;
    }
    all_pixel_data.push_back(pixels);
  }
  file.close();

  // 2. Barajar y tomar una fraccion aleatoria de los datos.
  size_t total_samples = all_labels.size();
  std::vector<size_t> indices(total_samples);
  std::iota(indices.begin(), indices.end(), 0);

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  std::random_shuffle(indices.begin(), indices.end());

  size_t samples_to_load = static_cast<size_t>(total_samples * sample_fraction);
  if (samples_to_load == 0 && total_samples > 0)
    samples_to_load = 1;
  if (samples_to_load > total_samples)
    samples_to_load = total_samples;

  std::vector<float> final_pixel_data;
  final_pixel_data.reserve(samples_to_load * 784);
  std::vector<int> final_labels;
  final_labels.reserve(samples_to_load);

  for (size_t i = 0; i < samples_to_load; ++i) {
    size_t original_index = indices[i];
    final_pixel_data.insert(final_pixel_data.end(), all_pixel_data[original_index].begin(),
                            all_pixel_data[original_index].end());
    final_labels.push_back(all_labels[original_index]);
  }

  // 3. Crear los tensores finales.
  // Forma de imagenes de entrada para ViT: {N, C, H, W}.
  Tensor X({samples_to_load, 1, 28, 28}, final_pixel_data);

  // Etiquetas en formato one-hot. MNIST/Fashion-MNIST tienen 10 clases.
  Tensor y = oneHotEncode(final_labels, 10);

  std::cout << "Carga completa. " << samples_to_load << " muestras cargadas." << std::endl;
  std::cout << "  -> Forma de X (imagenes): " << X.shapeToString() << std::endl;
  std::cout << "  -> Forma de y (etiquetas): " << y.shapeToString() << std::endl;

  return {std::move(X), std::move(y)};
}
