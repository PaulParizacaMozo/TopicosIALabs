
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
