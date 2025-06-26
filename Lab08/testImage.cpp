#include "CapaConvolucional.hpp"
#include "CapaFlatten.hpp"
#include "CapaPooling.hpp"
#include "Loader.hpp"
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
  try {
    // --- ETAPA 0: CARGAR IMAGEN ---
    cout << "--- ETAPA 0: CARGANDO IMAGEN ---" << endl;
    LoaderImage loader;
    Matriz imagenEntrada = loader.cargar("homer.jpg");

    cout << "\n--- MATRIZ DE ENTRADA (IMAGEN) ---" << endl;
    imagenEntrada.imprimir();

    int profundidadEntrada = imagenEntrada.getProfundidad();
    if (profundidadEntrada != 1 && profundidadEntrada != 3 && profundidadEntrada != 4) {
      cout << "Advertencia: La imagen tiene " << profundidadEntrada << " canales." << endl;
    }

    // --- ETAPA 1: CAPA CONVOLUCIONAL + ACTIVACION ReLU ---
    cout << "\n--- ETAPA 1: CAPA CONVOLUCIONAL + ReLU ---" << endl;
    const int NUMERO_DE_FILTROS = 3;
    const int TAMANO_FILTRO = 3;
    const int STRIDE = 1;
    const int PADDING = 0;

    CapaConvolucional capaConv(NUMERO_DE_FILTROS, TAMANO_FILTRO, profundidadEntrada, STRIDE, PADDING, TipoActivacion::RELU);

    cout << "\n--- FILTROS ALEATORIOS GENERADOS ---" << endl;
    capaConv.imprimirFiltros();

    Matriz mapaConv = capaConv.convolucion(imagenEntrada);

    cout << "\n--- SALIDA DE CAPA CONV + ReLU ---" << endl;
    mapaConv.imprimir();

    // --- ETAPA 2: CAPA MAX POOLING ---
    cout << "\n--- ETAPA 2: MAX POOLING ---" << endl;
    CapaPooling capaPool(TipoPooling::MAX, 2, 2);
    Matriz mapaPool = capaPool.aplicar(mapaConv);

    cout << "\n--- SALIDA DE CAPA POOLING ---" << endl;
    mapaPool.imprimir();

    // --- ETAPA 3: CAPA FLATTEN ---
    cout << "\n--- ETAPA 3: FLATTEN ---" << endl;
    vector<double> vectorAplanado = CapaFlatten::aplicar(mapaPool);

    cout << "\n--- SALIDA DE CAPA FLATTEN ---" << endl;
    cout << "Tamano total del vector: " << vectorAplanado.size() << endl;
    cout << "Vector: \n";
    imprimirVector(vectorAplanado, -1);

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}
