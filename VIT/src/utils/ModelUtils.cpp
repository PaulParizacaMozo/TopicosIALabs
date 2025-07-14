#include "utils/ModelUtils.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace ModelUtils {

void save_weights(const VisionTransformer &model, const std::string &filePath) {
  std::ofstream outFile(filePath, std::ios::binary);
  if (!outFile) {
    throw std::runtime_error("No se pudo abrir el archivo para escritura: " + filePath);
  }

  // Se usa const_cast para llamar a getParameters() en un modelo const.
  // Es seguro porque sabemos que getParameters() no modifica el modelo.
  VisionTransformer &non_const_model = const_cast<VisionTransformer &>(model);
  auto params = non_const_model.getParameters();

  std::cout << "Guardando " << params.size() << " tensores de parametros en " << filePath << "..." << std::endl;

  for (const auto &tensor_ptr : params) {
    const Tensor &tensor = *tensor_ptr;
    const auto &shape = tensor.getShape();
    size_t rank = shape.size();
    size_t num_elements = tensor.getSize();

    // 1. Escribir el numero de dimensiones (rank).
    outFile.write(reinterpret_cast<const char *>(&rank), sizeof(size_t));

    // 2. Escribir las dimensiones de la forma.
    outFile.write(reinterpret_cast<const char *>(shape.data()), rank * sizeof(size_t));

    // 3. Escribir los datos del tensor.
    // Si el tensor no es contiguo, se crea una copia temporal para guardar.
    if (!tensor.isContiguous()) {
      std::cerr << "Advertencia: Guardando un tensor no contiguo. Se creara una copia temporal." << std::endl;
      Tensor temp = tensor.contiguous();
      outFile.write(reinterpret_cast<const char *>(temp.getData()), num_elements * sizeof(float));
    } else {
      // Se accede a los datos considerando el offset por si es una vista.
      outFile.write(reinterpret_cast<const char *>(tensor.getData() + tensor.getDataOffset()), num_elements * sizeof(float));
    }
  }

  outFile.close();
  std::cout << "Pesos guardados correctamente." << std::endl;
}

void load_weights(VisionTransformer &model, const std::string &filePath) {
  std::ifstream inFile(filePath, std::ios::binary);
  if (!inFile) {
    throw std::runtime_error("No se pudo abrir el archivo para lectura: " + filePath);
  }

  auto params = model.getParameters();
  std::cout << "Cargando " << params.size() << " tensores de parametros desde " << filePath << "..." << std::endl;

  for (auto &tensor_ptr : params) {
    Tensor &tensor = *tensor_ptr;

    // Lee la forma del tensor desde el archivo.
    size_t file_rank;
    inFile.read(reinterpret_cast<char *>(&file_rank), sizeof(size_t));
    std::vector<size_t> file_shape(file_rank);
    inFile.read(reinterpret_cast<char *>(file_shape.data()), file_rank * sizeof(size_t));

    // Comprueba la compatibilidad de las formas.
    if (tensor.getShape() != file_shape) {
      throw std::runtime_error("Incompatibilidad de formas al cargar pesos. Esperado: " + tensor.shapeToString() +
                               ", encontrado: " + Tensor(file_shape).shapeToString());
    }

    size_t num_elements = tensor.getSize();
    // Cargar en un tensor no contiguo no esta implementado.
    if (!tensor.isContiguous()) {
      throw std::runtime_error("Cargar pesos a un tensor no contiguo no esta implementado.");
    } else {
      // Lee los datos directamente en la memoria del tensor.
      inFile.read(reinterpret_cast<char *>(tensor.getData() + tensor.getDataOffset()), num_elements * sizeof(float));
    }

    if (static_cast<size_t>(inFile.gcount()) != num_elements * sizeof(float)) {
      throw std::runtime_error("Error de lectura: fin de archivo inesperado o datos corruptos.");
    }
  }

  inFile.close();
  std::cout << "Pesos cargados correctamente." << std::endl;
}

} // namespace ModelUtils
