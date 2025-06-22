#include "Matriz.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

Matriz::Matriz(int w, int h, int d) : ancho(w), alto(h), profundidad(d) {
  // Inicializamos con 0
  datos.resize(d, vector<vector<double>>(h, vector<double>(w, 0.0)));
}

Matriz::Matriz() : ancho(0), alto(0), profundidad(0) {}

int Matriz::getAncho() const { return ancho; }
int Matriz::getAlto() const { return alto; }
int Matriz::getProfundidad() const { return profundidad; }

void Matriz::establecerValor(int x, int y, int z, double valor) {
  if (z >= 0 && z < profundidad && y >= 0 && y < alto && x >= 0 && x < ancho) {
    datos[z][y][x] = valor;
  }
}

double Matriz::obtenerValor(int x, int y, int z) const {
  if (z >= 0 && z < profundidad && y >= 0 && y < alto && x >= 0 && x < ancho) {
    return datos[z][y][x];
  }
  return 0.0; // si esta fuera de rango
}

void Matriz::imprimir() const {
  cout << "Dimensiones de la Matriz: " << getAncho() << "x" << getAlto() << "x" << getProfundidad() << endl;

  for (int z = 0; z < profundidad; ++z) {
    cout << "--- Canal " << z << " ---" << endl;
    for (int y = 0; y < alto; ++y) {
      for (int x = 0; x < ancho; ++x) {
        cout << fixed << setprecision(2) << setw(8) << datos[z][y][x];
      }
      cout << endl;
    }
    cout << endl;
  }
}
