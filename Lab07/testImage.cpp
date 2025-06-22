#include "CapaConvolucional.hpp"
#include "Loader.hpp"
#include <iostream>
using namespace std;

int main() {
  try {
    LoaderImage loader;
    Matriz imagenEntrada = loader.cargar("homer.jpg");

    int profundidadEntrada = imagenEntrada.getProfundidad();
    if (profundidadEntrada != 1 && profundidadEntrada != 3) {
      cout << "La imagen tiene " << profundidadEntrada << " canales." << endl;
    }

    const int NUMERO_DE_FILTROS = 3;
    const int TAMANO_FILTRO = 3;
    const int STRIDE = 1;
    const int PADDING = 1;

    CapaConvolucional capa(NUMERO_DE_FILTROS, TAMANO_FILTRO, profundidadEntrada, STRIDE, PADDING);

    cout << "\n--- FILTROS ---" << endl;
    capa.imprimirFiltros();

    // 4. Aplicar la convolucion
    cout << "\n--- APLICANDO CONVOLUCION... ---" << endl;
    Matriz mapaDeCaracteristicas = capa.convolucion(imagenEntrada);

    // 5. Mostrar el resultado
    cout << "\n--- MAPA DE CARACTERISTICAS (SALIDA) ---" << endl;
    mapaDeCaracteristicas.imprimir();

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}
