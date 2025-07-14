#include "model/Trainer.hpp"
#include "utils/DataReader.hpp"
#include "utils/ModelUtils.hpp"
#include <iostream>

// Punto de entrada principal de la aplicacion.
int main() {
  try {
    // --- 1. Definir Configuraciones del Modelo y Entrenador ---
    ViTConfig model_config;
    // Hiperparametros de la arquitectura del Vision Transformer.
    model_config.embedding_dim = 64;
    model_config.num_layers = 1;
    model_config.num_heads = 2;
    model_config.mlp_hidden_dim = model_config.embedding_dim * 4;

    TrainerConfig train_config;
    // Hiperparametros para el bucle de entrenamiento.
    train_config.epochs = 10;
    train_config.batch_size = 32;
    train_config.learning_rate = 0.0001f;
    train_config.weight_decay = 0.01f;

    // --- 2. Cargar los datos de entrenamiento y prueba desde archivos CSV ---
    std::cout << "--- Cargando Datos de Fashion MNIST ---" << std::endl;
    auto train_data = load_csv_data("data/fashion_train.csv", 1.0f);
    auto test_data = load_csv_data("data/fashion_test.csv", 1.0f);

    // --- 3. Crear la instancia del modelo y pasarla al entrenador ---
    VisionTransformer model(model_config);
    Trainer trainer(model, train_config);

    // --- 4. Iniciar el bucle de entrenamiento y evaluacion ---
    trainer.train(train_data, test_data);

    std::cout << "\n¡Entrenamiento completado!" << std::endl;

    // --- 5. Guardar los pesos del modelo entrenado para uso futuro ---
    const std::string weights_path = "vit_fashion_mnist.weights.1_2_64_32";
    std::cout << "\nGuardando pesos del modelo entrenado en: " << weights_path << std::endl;
    ModelUtils::save_weights(model, weights_path);

    std::cout << "\nProceso finalizado." << std::endl;

  } catch (const std::exception &e) {
    // Captura cualquier excepcion estandar y la imprime en stderr.
    std::cerr << "\nERROR CRITICO: " << e.what() << std::endl;
    return 1; // Devuelve 1 en caso de error.
  }

  return 0; // Devuelve 0 si todo fue exitoso.
}
