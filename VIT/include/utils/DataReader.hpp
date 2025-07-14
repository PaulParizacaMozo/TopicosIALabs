#ifndef DATAREADER_HPP
#define DATAREADER_HPP

#include "core/Tensor.hpp"
#include <string>
#include <utility>

// Carga y procesa un dataset tipo MNIST/Fashion-MNIST desde un archivo CSV.
// Detalles:
// - Lee un CSV donde la primera columna es la etiqueta y las siguientes son pixeles.
// - Normaliza los valores de los pixeles al rango [0, 1].
// - Codifica las etiquetas en formato one-hot.
// - Remodela los datos a la forma de imagen 4D {N, C, H, W}.
//
// - filePath: Ruta al archivo .csv.
// - sample_fraction: Fraccion del dataset a cargar (de 0.0 a 1.0).
// Devuelve un par de Tensores: {Imagenes, Etiquetas}.
std::pair<Tensor, Tensor> load_csv_data(const std::string &filePath, float sample_fraction = 1.0f);

#endif // DATAREADER_HPP
