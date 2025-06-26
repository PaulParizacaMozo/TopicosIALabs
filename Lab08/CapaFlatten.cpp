#include "CapaFlatten.hpp"

std::vector<double> CapaFlatten::aplicar(const Matriz &entrada) {
  int ancho = entrada.getAncho();
  int alto = entrada.getAlto();
  int profundidad = entrada.getProfundidad();

  int tamanoTotal = ancho * alto * profundidad;
  std::vector<double> vectorAplanado;
  vectorAplanado.reserve(tamanoTotal);

  for (int z = 0; z < profundidad; ++z) {
    for (int y = 0; y < alto; ++y) {
      for (int x = 0; x < ancho; ++x) {
        vectorAplanado.push_back(entrada.obtenerValor(x, y, z));
      }
    }
  }

  return vectorAplanado;
}
