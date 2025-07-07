// app/main.cpp

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility> // Para std::pair
#include <vector>

// --- Incluir nuestros componentes de la librería ---
#include "activations/ReLU.hpp"
#include "core/Tensor.hpp"
#include "layers/Conv2D.hpp"
#include "layers/Dense.hpp"
#include "layers/Dropout.hpp"
#include "layers/Flatten.hpp"
#include "layers/Pooling2D.hpp"
#include "losses/CrossEntropy.hpp"
#include "model/Sequential.hpp"
#include "optimizers/Adam.hpp"
#include "optimizers/SGD.hpp"
#include <omp.h>

// --- DECLARACIÓN ANTICIPADA ---
// Le decimos al compilador que la función loadMnist existe, que toma estos
// argumentos y devuelve este tipo. La implementación real la encontrará
// el enlazador (linker) en DataReader.o.
std::pair<Tensor, Tensor> loadMnist(const std::string &filePath, float sampleFraction = 1.0f, int channels = 1,
                                    bool shuffle = true);

void loadModel(Sequential &model, const std::string &filePath);
void saveModel(const Sequential &model, const std::string &filePath);

void predictAndDraw(Sequential &model, const Tensor &X_test, const Tensor &y_test, size_t index);
// Función para obtener el índice de la clase con la probabilidad más alta
// size_t argmax(const float *data, size_t size) { return std::distance(data, std::max_element(data, data + size)); }

int main() {
  try {
    // --- Definir la ruta del archivo del modelo ---
    const std::string modelFilePath = "data/cnn_3canales_20.bin";

    // --- 1. Definir la Arquitectura del Modelo ---
    // Es importante definirla primero, ya que 'loadModel' la necesita.
    Sequential model;
    model.add<Conv2D>(3, 16, 3, 1, 1);
    model.add<ReLU>();
    model.add<Pooling2D>(2);
    model.add<Conv2D>(16, 4, 3, 1, 1);
    model.add<ReLU>();
    model.add<Pooling2D>(2);
    model.add<Flatten>();
    model.add<Dense>(196, 16);
    model.add<ReLU>();
    model.add<Dense>(16, 10);
    /*
    model.add<Flatten>();
    model.add<Dense>(784, 128);
    model.add<ReLU>();
    model.add<Dense>(128, 64);
    model.add<ReLU>();
    model.add<Dense>(64, 10);
    */

    // --- 2. Cargar Datos ---
    auto [X_train, y_train] = loadMnist("data/fashion_train.csv", 1.0f, 3); // Entrenamos con 20%
    auto [X_test, y_test] = loadMnist("data/fashion_test.csv", 1.0f, 3);

    // --- 3. Compilar y Entrenar ---
    model.compile<SGD, CrossEntropy>(0.002f);

    std::cout << "\n--- Iniciando Entrenamiento de la CNN ---\n" << std::endl;
    model.train(X_train, y_train, 20, 8, X_test, y_test);
    std::cout << "\n--- Entrenamiento Finalizado ---\n" << std::endl;

    // --- 4. Guardar el Modelo Entrenado ---
    saveModel(model, modelFilePath);

    // --- 5. Opcional: Probar la carga y re-evaluar ---
    std::cout << "\n--- Probando la carga del modelo ---" << std::endl;

    // Creamos una nueva instancia del modelo con la MISMA arquitectura.
    Sequential loadedModel;
    loadedModel.add<Conv2D>(3, 16, 3, 1, 1);
    loadedModel.add<ReLU>();
    loadedModel.add<Pooling2D>(2);
    loadedModel.add<Conv2D>(16, 4, 3, 1, 1);
    loadedModel.add<ReLU>();
    loadedModel.add<Pooling2D>(2);
    loadedModel.add<Flatten>();
    loadedModel.add<Dense>(196, 16);
    loadedModel.add<ReLU>();
    loadedModel.add<Dense>(16, 10);
    /*
    loadedModel.add<Flatten>();
    loadedModel.add<Dense>(784, 128);
    loadedModel.add<ReLU>();
    loadedModel.add<Dense>(128, 64);
    loadedModel.add<ReLU>();
    loadedModel.add<Dense>(64, 10);

    */
    // Compilamos el nuevo modelo (necesario para tener la función de pérdida)
    loadedModel.compile<SGD, CrossEntropy>(0.002f);

    // Cargamos los pesos guardados
    loadModel(loadedModel, modelFilePath);

    // Evaluamos el modelo cargado para confirmar que funciona
    std::cout << "\n--- Evaluando modelo CARGADO en el conjunto de prueba ---" << std::endl;
    auto [finalLoss, finalAccuracy] = loadedModel.evaluate(X_test, y_test);

    std::cout << "========================================" << std::endl;
    std::cout << "  Rendimiento del Modelo Cargado" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Perdida (Loss) en Test: " << finalLoss << std::endl;
    std::cout << "  Precision (Accuracy) en Test: " << finalAccuracy * 100.0f << "%" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==  NUEVO: Predecir y dibujar algunas muestras al azar ==

    // Inicializamos el generador de números aleatorios para elegir índices
    srand(static_cast<unsigned int>(time(0)));
    // Probamos 5 imágenes aleatorias del conjunto de prueba
    for (int i = 0; i < 1; ++i) {
      size_t randomIndex = rand() % X_test.getShape()[0];
      predictAndDraw(loadedModel, X_test, y_test, randomIndex);
    }

  } catch (const std::exception &e) {
    std::cerr << "Ha ocurrido un error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
