#include "CapaPooling.hpp"
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
