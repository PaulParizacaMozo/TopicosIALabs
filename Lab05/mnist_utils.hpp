#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// estructura para devolver los datos cargados
struct MNISTData {
  vector<vector<double>> entradas;         // Entradas de las muestras
  vector<vector<double>> salidasEsperadas; // Salidas esperadas (one-hot)
};

// func cargar los datos desde un archivo CSV
MNISTData cargarDatosCSV(const string &nombreArchivo, int numMuestrasACargar, int numClases = 10) {
  MNISTData datos;
  ifstream archivo(nombreArchivo);

  if (!archivo.is_open()) {
    cerr << "Error al abrir el archivo: " << nombreArchivo << endl;
    return datos;
  }

  string linea;
  // Ignorar la primera linea de encabezados
  if (getline(archivo, linea)) {
  } else {
    cerr << "Archivo CSV vacío o no se pudo leer el encabezado: " << nombreArchivo << endl;
    return datos;
  }

  int muestrasCargadas = 0;
  while (getline(archivo, linea) && (numMuestrasACargar == -1 || muestrasCargadas < numMuestrasACargar)) {
    vector<double> pixelesImagen;
    vector<double> etiquetaOneHot(numClases, 0.0);

    stringstream ss(linea);
    string valorCelda;

    // Leer la etiqueta (primer valor)
    if (getline(ss, valorCelda, ',')) {
      int etiqueta = stoi(valorCelda);
      if (etiqueta >= 0 && etiqueta < numClases) {
        etiquetaOneHot[etiqueta] = 1.0;
      } else {
        cerr << "Etiqueta fuera de rango: " << etiqueta << " en línea: " << linea << endl;
        continue;
      }
    } else {
      cerr << "Linea CSV mal formada (sin etiqueta): " << linea << endl;
      continue;
    }

    // Leer los 784 valores de pixeles
    while (getline(ss, valorCelda, ',')) {
      // Normalizar los valores de los píxeles a [0, 1]
      pixelesImagen.push_back(stod(valorCelda) / 255.0);
    }

    if (pixelesImagen.size() == 784) { // se leyeron todos los pixeles
      datos.entradas.push_back(pixelesImagen);
      datos.salidasEsperadas.push_back(etiquetaOneHot);
      muestrasCargadas++;
    } else if (!pixelesImagen.empty()) { //  no vacia pero no tiene 784, es un error
      cerr << "Numero incorrecto de píxeles" << endl;
    }
  }

  archivo.close();
  cout << "Cargadas " << muestrasCargadas << " muestras de " << nombreArchivo << endl;
  return datos;
}

#endif
