/*
 * Codigo completo: Se juntos todos los hpp y cpp de las clases
 * Se puede ejecutar con:
   g++ CodigoCompleto.cpp -o test
   ./test
*/

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

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace std;

// constructor
CapaConvolucional::CapaConvolucional(int numFiltros, int tamFiltro, int profFiltro, int s, int p, TipoActivacion act)
    : numeroDeFiltros(numFiltros), stride(s), padding(p), activacion(act) {
  inicializarFiltros(tamFiltro, profFiltro);
}

double CapaConvolucional::aplicarActivacion(double valor) {
  switch (activacion) {
  case TipoActivacion::RELU:
    return std::max(0.0, valor);
  case TipoActivacion::SIGMOID:
    return 1.0 / (1.0 + std::exp(-valor));
  case TipoActivacion::TANH:
    return std::tanh(valor);
  case TipoActivacion::NINGUNA:
  default:
    return valor;
  }
}

// initcializar filtros
void CapaConvolucional::inicializarFiltros(int tamanoFiltro, int profundidadFiltro) {
  double fan_in = tamanoFiltro * tamanoFiltro * profundidadFiltro;
  // desviacion estandar
  double stddev = sqrt(2.0 / fan_in);

  uint32_t semilla = 4444;
  mt19937 gen(semilla);
  // random_device rd;
  //  mt19937 gen(rd());
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
        // Fun activacion
        double valorActivado = aplicarActivacion(suma);
        mapaDeCaracteristicas.establecerValor(x, y, f, valorActivado);
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

#include <limits>
#include <stdexcept>

CapaPooling::CapaPooling(TipoPooling t, int ventana, int s, int p) : tipo(t), tamanoVentana(ventana), stride(s), padding(p) {
  if (tamanoVentana <= 0 || stride <= 0 || padding < 0) {
    throw std::invalid_argument("El tamano de la ventana, stride y padding deben ser no negativos (ventana y stride > 0).");
  }
}

Matriz CapaPooling::aplicar(const Matriz &entrada) {
  int anchoEntrada = entrada.getAncho();
  int altoEntrada = entrada.getAlto();
  int profundidadEntrada = entrada.getProfundidad();

  // pooling - padding + stride
  int anchoSalida = (anchoEntrada - tamanoVentana + 2 * padding) / stride + 1;
  int altoSalida = (altoEntrada - tamanoVentana + 2 * padding) / stride + 1;

  // misma profuncidad en salida
  Matriz salida(anchoSalida, altoSalida, profundidadEntrada);

  Matriz entradaConPadding(anchoEntrada + 2 * padding, altoEntrada + 2 * padding, profundidadEntrada);
  for (int z = 0; z < profundidadEntrada; ++z) {
    for (int y = 0; y < altoEntrada; ++y) {
      for (int x = 0; x < anchoEntrada; ++x) {
        entradaConPadding.establecerValor(x + padding, y + padding, z, entrada.obtenerValor(x, y, z));
      }
    }
  }

  // it sobre cada canal
  for (int z = 0; z < profundidadEntrada; ++z) {
    for (int y = 0; y < altoSalida; ++y) {
      for (int x = 0; x < anchoSalida; ++x) {

        double valorResultante = 0.0;
        int y_inicio = y * stride;
        int x_inicio = x * stride;

        if (tipo == TipoPooling::MAX) {
          valorResultante = -std::numeric_limits<double>::infinity();
        }

        double sumaPromedio = 0.0;

        // it sobre la ventana de pooling
        for (int vy = 0; vy < tamanoVentana; ++vy) {
          for (int vx = 0; vx < tamanoVentana; ++vx) {
            double valorActual = entradaConPadding.obtenerValor(x_inicio + vx, y_inicio + vy, z);

            if (tipo == TipoPooling::MAX) {
              if (valorActual > valorResultante) {
                valorResultante = valorActual;
              }
            } else { // AVERAGE
              sumaPromedio += valorActual;
            }
          }
        }

        if (tipo == TipoPooling::AVERAGE) {
          valorResultante = sumaPromedio / (tamanoVentana * tamanoVentana);
        }

        salida.establecerValor(x, y, z, valorResultante);
      }
    }
  }
  return salida;
}

#include <vector>

class CapaFlatten {
public:
  static std::vector<double> aplicar(const Matriz &entrada);
};

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

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

void imprimirVector(const vector<double> &vec, int elementos) {
  if (vec.empty()) {
    cout << "[]" << endl;
    return;
  }

  int limite = (elementos == -1) ? vec.size() : min((int)vec.size(), elementos);

  cout << "[";
  for (int i = 0; i < limite; ++i) {
    cout << vec[i];
    if (i < limite - 1) {
      cout << ", ";
    }
  }

  if (elementos != -1 && vec.size() > elementos) {
    cout << " ...";
  }

  cout << "]" << endl;
}

int main() {
  // --- CONFIGURACION INICIAL ---
  const int ANCHO_ENTRADA = 8;
  const int ALTO_ENTRADA = 8;
  const int PROFUNDIDAD_ENTRADA = 3;

  // Matriz de entrada (8x8x3)
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

  // --- CAPA CONVOLUCIONAL + FUNCION SIGMOIDE ---

  const int NUMERO_DE_FILTROS = 4;
  const int TAMANO_FILTRO = 3;
  const int STRIDE_CONV = 1;
  const int PADDING_CONV = 1;

  // creamos capa convolucional
  CapaConvolucional capaConv(NUMERO_DE_FILTROS, TAMANO_FILTRO, PROFUNDIDAD_ENTRADA, STRIDE_CONV, PADDING_CONV,
                             TipoActivacion::SIGMOID);

  cout << "*** FILTROS ALEATORIOS ***" << endl;
  capaConv.imprimirFiltros();

  cout << "*** ETAPA 1: CONVOLUCION + FUNCION SIGMOIDE ***" << endl;
  // Aplicamos la convolucion
  Matriz mapaConv = capaConv.convolucion(imagenEntrada);

  cout << "\n***  SALIDA DE CAPA CONV + SIGMOIDE  ***" << endl;
  mapaConv.imprimir();

  // --- CAPA DE POOLING ---
  cout << "***  ETAPA 2: AVG POOLING  ***" << endl;

  // capa de pooling con una ventana 2x2 y stride 2
  CapaPooling capaPool(TipoPooling::AVERAGE, 2, 2);
  Matriz mapaPool = capaPool.aplicar(mapaConv);

  cout << "***  SALIDA DE CAPA POOLING ***" << endl;
  mapaPool.imprimir();

  // --- CAPA FLATTEN ---
  cout << "***  ETAPA 3: FLATTEN  ***" << endl;

  // aplanamiento
  vector<double> vectorAplanado = CapaFlatten::aplicar(mapaPool);

  cout << "***  SALIDA DE CAPA FLATTEN  ***" << endl;
  cout << "Tamano total del vector: " << vectorAplanado.size() << endl;
  cout << "Vector aplanado: ";
  imprimirVector(vectorAplanado, -1);

  return 0;
}
