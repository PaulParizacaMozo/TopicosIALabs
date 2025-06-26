#ifndef LOADER_HPP
#define LOADER_HPP

#include "Matriz.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

class LoaderImage {
public:
  Matriz cargar(const string &rutaArchivo) {
    int ancho, alto, canalesOriginales;

    // stbi_load carga la imagen y devuelve un puntero a los datos de los pixeles.
    // array de unsigned char (valores de 0 a 255).
    unsigned char *datosImagen = stbi_load(rutaArchivo.c_str(), &ancho, &alto, &canalesOriginales, 0);

    if (datosImagen == nullptr) {
      throw runtime_error("Error al cargar la imagen: " + rutaArchivo + ". Razon: " + stbi_failure_reason());
    }

    cout << "Imagen cargada exitosamente: " << rutaArchivo << endl;
    cout << "Dimensiones: " << ancho << "x" << alto << ", Canales: " << canalesOriginales << endl;

    // matriz de entrada
    Matriz matrizSalida(ancho, alto, canalesOriginales);

    for (int y = 0; y < alto; ++y) {
      for (int x = 0; x < ancho; ++x) {
        for (int z = 0; z < canalesOriginales; ++z) {
          // estructura es [R1, G1, B1, R2, G2, B2, ...]
          int indice = (y * ancho + x) * canalesOriginales + z;
          unsigned char valorPixel = datosImagen[indice];

          // normalizar
          double valorNormalizado = static_cast<double>(valorPixel) / 255.0;

          matrizSalida.establecerValor(x, y, z, valorNormalizado);
        }
      }
    }
    // liberar memoria
    stbi_image_free(datosImagen);

    return matrizSalida;
  }
};

#endif
