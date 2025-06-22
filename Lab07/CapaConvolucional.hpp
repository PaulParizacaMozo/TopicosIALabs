#ifndef CAPA_CONVOLUCIONAL_HPP
#define CAPA_CONVOLUCIONAL_HPP

#include "Matriz.hpp"
#include <vector>
using namespace std;

class CapaConvolucional {
private:
  int numeroDeFiltros;
  int stride;
  int padding;
  vector<Matriz> filtros;

  // inicializar filtros
  void inicializarFiltros(int tamanoFiltro, int profundidadFiltro);

public:
  // constructor
  CapaConvolucional(int numeroDeFiltros, int tamanoFiltro, int profundidadFiltro, int stride, int padding);

  // convolucion
  Matriz convolucion(const Matriz &entrada);

  void imprimirFiltros() const;
};

#endif
