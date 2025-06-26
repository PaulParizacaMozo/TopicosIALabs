#ifndef CAPA_POOLING_HPP
#define CAPA_POOLING_HPP

#include "Matriz.hpp"

// Enum para seleccionar el tipo de pooling
enum class TipoPooling { MAX, AVERAGE };

class CapaPooling {
private:
  TipoPooling tipo;
  int tamanoVentana;
  int stride;
  int padding;

public:
  // Constructor
  CapaPooling(TipoPooling tipo, int tamanoVentana, int stride, int padding = 0);

  // Metodo para aplicar el pooling a un mapa de caracteristicas
  Matriz aplicar(const Matriz &entrada);
};

#endif
