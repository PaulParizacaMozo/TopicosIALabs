#include "CapaConvolucional.hpp"
#include "CapaFlatten.hpp"
#include "CapaPooling.hpp"
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
