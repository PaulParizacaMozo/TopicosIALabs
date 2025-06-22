#include "CapaConvolucional.hpp"
#include <iostream>

int main() {
  const int ANCHO_ENTRADA = 8;
  const int ALTO_ENTRADA = 8;
  const int PROFUNDIDAD_ENTRADA = 3;

  const int NUMERO_DE_FILTROS = 4;
  const int TAMANO_FILTRO = 3;
  const int STRIDE = 1;
  const int PADDING = 1;

  // matriz de entrada(8x8x3)
  Matriz imagenEntrada(ANCHO_ENTRADA, ALTO_ENTRADA, PROFUNDIDAD_ENTRADA);
  for (int z = 0; z < PROFUNDIDAD_ENTRADA; ++z) {
    for (int y = 0; y < ALTO_ENTRADA; ++y) {
      for (int x = 0; x < ANCHO_ENTRADA; ++x) {
        imagenEntrada.establecerValor(x, y, z, x + y + z);
      }
    }
  }

  cout << "***     IMAGEN DE ENTRADA       ***" << endl;
  imagenEntrada.imprimir();

  CapaConvolucional capa(NUMERO_DE_FILTROS, TAMANO_FILTRO, PROFUNDIDAD_ENTRADA, STRIDE, PADDING);

  cout << "***      FILTROS***" << endl;
  capa.imprimirFiltros();

  Matriz mapaDeCaracteristicas = capa.convolucion(imagenEntrada);

  cout << "***  MAPA DE CARACTERISTICAS    ***" << endl;
  mapaDeCaracteristicas.imprimir();

  // prueba con stride = 2 y padding = 0
  cout << "\n--- Prueba con Stride = 2 y Padding = 0 (valid) ---" << endl;
  CapaConvolucional capa2(2, 3, 3, 2, 0);
  Matriz mapaReducido = capa2.convolucion(imagenEntrada);
  cout << "***  MAPA DE CARACTERISTICAS    ***" << endl;
  mapaReducido.imprimir();

  return 0;
}
