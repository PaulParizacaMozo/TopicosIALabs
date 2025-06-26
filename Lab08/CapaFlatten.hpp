#ifndef CAPA_FLATTEN_HPP
#define CAPA_FLATTEN_HPP

#include "Matriz.hpp"
#include <vector>

class CapaFlatten {
public:
  static std::vector<double> aplicar(const Matriz &entrada);
};

#endif
