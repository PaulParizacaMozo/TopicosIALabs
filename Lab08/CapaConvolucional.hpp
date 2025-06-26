#ifndef CAPA_CONVOLUCIONAL_HPP
#define CAPA_CONVOLUCIONAL_HPP

#include "Matriz.hpp"
#include <vector>
using namespace std;

enum class TipoActivacion { NINGUNA, RELU, SIGMOID, TANH };

class CapaConvolucional {
private:
  int numeroDeFiltros;
  int stride;
  int padding;
  TipoActivacion activacion;
  vector<Matriz> filtros;

  // inicializar filtros
  void inicializarFiltros(int tamanoFiltro, int profundidadFiltro);

  // Metodo privado para aplicar la funcion de activacion seleccionada
  double aplicarActivacion(double valor);

public:
  // constructor
  CapaConvolucional(int numeroDeFiltros, int tamanoFiltro, int profundidadFiltro, int stride, int padding,
                    TipoActivacion activacion = TipoActivacion::NINGUNA);

  // convolucion
  Matriz convolucion(const Matriz &entrada);

  void imprimirFiltros() const;
};

#endif
