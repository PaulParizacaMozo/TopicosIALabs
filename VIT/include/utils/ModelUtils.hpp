#ifndef MODELUTILS_HPP
#define MODELUTILS_HPP

#include "model/VisionTransformer.hpp"
#include <string>
#include <vector>

namespace ModelUtils {

// Guarda los parametros (pesos) de un modelo en un archivo binario.
// Formato por tensor:
//   1. (size_t) Numero de dimensiones.
//   2. (size_t*) Dimensiones de la forma.
//   3. (float*) Datos del tensor.
void save_weights(const VisionTransformer &model, const std::string &filePath);

// Carga los pesos desde un archivo binario a un modelo existente.
// El modelo debe tener la misma arquitectura (formas de tensor) que el guardado.
void load_weights(VisionTransformer &model, const std::string &filePath);

} // namespace ModelUtils

#endif // MODELUTILS_HPP
