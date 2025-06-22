#include "CapaConvolucional.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace std;

// constructor
CapaConvolucional::CapaConvolucional(int numFiltros, int tamFiltro, int profFiltro, int s, int p)
    : numeroDeFiltros(numFiltros), stride(s), padding(p) {
  inicializarFiltros(tamFiltro, profFiltro);
}

// initcializar filtros
void CapaConvolucional::inicializarFiltros(int tamanoFiltro, int profundidadFiltro) {
  double fan_in = tamanoFiltro * tamanoFiltro * profundidadFiltro;
  // desviacion estandar
  double stddev = sqrt(2.0 / fan_in);

  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> distribucionHe(0.0, stddev);

  filtros.clear();
  for (int i = 0; i < numeroDeFiltros; ++i) {
    Matriz filtro(tamanoFiltro, tamanoFiltro, profundidadFiltro);
    for (int z = 0; z < profundidadFiltro; ++z) {
      for (int y = 0; y < tamanoFiltro; ++y) {
        for (int x = 0; x < tamanoFiltro; ++x) {
          filtro.establecerValor(x, y, z, distribucionHe(gen));
        }
      }
    }
    filtros.push_back(filtro);
  }
}

Matriz CapaConvolucional::convolucion(const Matriz &entrada) {
  if (entrada.getProfundidad() != filtros[0].getProfundidad()) {
    throw invalid_argument("Error: La profundidad de la matriz de entrada no coincide con la profundidad de los filtros.");
  }

  int tamanoFiltro = filtros[0].getAncho();

  // dimensiones de la salida
  int anchoSalida = (entrada.getAncho() - tamanoFiltro + 2 * padding) / stride + 1;
  int altoSalida = (entrada.getAlto() - tamanoFiltro + 2 * padding) / stride + 1;
  int profundidadSalida = numeroDeFiltros;

  // matriz de salida
  Matriz mapaDeCaracteristicas(anchoSalida, altoSalida, profundidadSalida);

  // matriz de entrada + padding
  Matriz entradaConPadding(entrada.getAncho() + 2 * padding, entrada.getAlto() + 2 * padding, entrada.getProfundidad());
  for (int z = 0; z < entrada.getProfundidad(); ++z) {
    for (int y = 0; y < entrada.getAlto(); ++y) {
      for (int x = 0; x < entrada.getAncho(); ++x) {
        entradaConPadding.establecerValor(x + padding, y + padding, z, entrada.obtenerValor(x, y, z));
      }
    }
  }

  // convolucion
  // it cada filtro
  for (int f = 0; f < numeroDeFiltros; ++f) {
    // it eje y
    for (int y = 0; y < altoSalida; ++y) {
      // it eje x
      for (int x = 0; x < anchoSalida; ++x) {

        double suma = 0.0;

        // producto punto
        for (int c = 0; c < filtros[f].getProfundidad(); ++c) { // canal
          for (int fy = 0; fy < tamanoFiltro; ++fy) {           // Y del filtro
            for (int fx = 0; fx < tamanoFiltro; ++fx) {         // X del filtro
              int coordYEntrada = y * stride + fy;
              int coordXEntrada = x * stride + fx;

              double valorEntrada = entradaConPadding.obtenerValor(coordXEntrada, coordYEntrada, c);
              double valorFiltro = filtros[f].obtenerValor(fx, fy, c);

              suma += valorEntrada * valorFiltro;
            }
          }
        }

        mapaDeCaracteristicas.establecerValor(x, y, f, suma);
      }
    }
  }

  return mapaDeCaracteristicas;
}

void CapaConvolucional::imprimirFiltros() const {
  cout << "--- Mostrando " << numeroDeFiltros << " Filtros ---" << endl;
  for (int i = 0; i < filtros.size(); ++i) {
    cout << "\n=== Filtro " << i << " ===" << endl;
    filtros[i].imprimir();
  }
}
