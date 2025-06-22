#ifndef MATRIZ_HPP
#define MATRIZ_HPP

#include <vector>
using namespace std;

// matriz 3D
class Matriz {
private:
  int ancho;
  int alto;
  int profundidad;
  vector<vector<vector<double>>> datos;

public:
  Matriz(int ancho, int alto, int profundidad);

  Matriz();

  // Getters
  int getAncho() const;
  int getAlto() const;
  int getProfundidad() const;

  void establecerValor(int x, int y, int z, double valor);
  double obtenerValor(int x, int y, int z) const;

  void imprimir() const;
};

#endif
