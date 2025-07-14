#include "model/Trainer.hpp"
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

// --- Funcion Auxiliar (privada a este archivo) ---
namespace {
// Calcula la precision de las predicciones de un batch.
float calculate_accuracy(const Tensor &logits, const Tensor &labels) {
  size_t batch_size = logits.getShape()[0];
  if (batch_size == 0)
    return 0.0f;
  size_t correct_predictions = 0;

  for (size_t i = 0; i < batch_size; ++i) {
    // Encuentra el indice de la clase con la mayor puntuacion (argmax).
    float max_logit = -std::numeric_limits<float>::infinity();
    int pred_class = -1;
    for (size_t j = 0; j < logits.getShape()[1]; ++j) {
      if (logits(i, j) > max_logit) {
        max_logit = logits(i, j);
        pred_class = j;
      }
    }
    // Comprueba si coincide con la etiqueta verdadera (que es 1.0 en one-hot).
    if (labels(i, pred_class) == 1.0f) {
      correct_predictions++;
    }
  }
  return static_cast<float>(correct_predictions) / batch_size;
}
} // namespace

// Constructor del Trainer. Recibe una referencia al modelo y la configuracion.
Trainer::Trainer(VisionTransformer &model, const TrainerConfig &train_config)
    : model(model), optimizer(train_config.learning_rate, 0.9f, 0.999f, 1e-8f, train_config.weight_decay), loss_fn(),
      config(train_config) {}

// Orquesta el proceso de entrenamiento completo a lo largo de varias epocas.
void Trainer::train(const std::pair<Tensor, Tensor> &train_data, const std::pair<Tensor, Tensor> &test_data) {
  const auto &[X_train, y_train] = train_data;
  const auto &[X_test, y_test] = test_data;

  for (int epoch = 0; epoch < config.epochs; ++epoch) {
    // Ejecuta una epoca de entrenamiento y obtiene sus metricas.
    auto [train_loss, train_acc] = train_epoch(X_train, y_train);

    // Limpia la linea de progreso de los batches.
    std::cout << "\r" << std::string(80, ' ') << "\r";

    // Evalua en el conjunto de test para obtener sus metricas.
    auto [test_loss, test_acc] = evaluate(X_test, y_test);

    // Imprime el resumen de la epoca.
    std::cout << "--- Epoca " << epoch + 1 << "/" << config.epochs << " | Train Loss: " << std::fixed << std::setprecision(4)
              << train_loss << " | Train Acc: " << train_acc << " | Test Loss: " << test_loss << " | Test Acc: " << test_acc
              << std::endl;
  }
}

// Ejecuta un ciclo completo sobre el dataset de entrenamiento (una epoca).
std::pair<float, float> Trainer::train_epoch(const Tensor &X_train, const Tensor &y_train) {
  size_t num_train_samples = X_train.getShape()[0];
  size_t num_batches = (num_train_samples + config.batch_size - 1) / config.batch_size;

  float total_loss = 0.0f;
  float total_accuracy = 0.0f;

  // Crea y baraja los indices para procesar los datos en orden aleatorio.
  std::vector<size_t> indices(num_train_samples);
  std::iota(indices.begin(), indices.end(), 0);
  std::srand(static_cast<unsigned int>(std::time(nullptr))); // Semilla para el barajado.
  std::random_shuffle(indices.begin(), indices.end());

  for (size_t i = 0; i < num_batches; ++i) {
    size_t start_idx = i * config.batch_size;
    size_t count = std::min(config.batch_size, num_train_samples - start_idx);
    if (count == 0)
      continue;

    // Crea los tensores para el batch actual.
    // Nota: Esta seccion podria optimizarse creando una funcion 'batch_slice'
    // que extraiga un batch de datos usando una lista de indices.
    Tensor X_batch({count, X_train.getShape()[1], X_train.getShape()[2], X_train.getShape()[3]});
    Tensor y_batch({count, y_train.getShape()[1]});

    for (size_t j = 0; j < count; ++j) {
      size_t data_idx = indices[start_idx + j];
      Tensor x_sample = X_train.slice(0, data_idx, 1);
      Tensor y_sample = y_train.slice(0, data_idx, 1);
      // Copia manual de datos.
      for (size_t c = 0; c < X_batch.getShape()[1]; ++c)
        for (size_t h = 0; h < X_batch.getShape()[2]; ++h)
          for (size_t w = 0; w < X_batch.getShape()[3]; ++w)
            X_batch(j, c, h, w) = x_sample(0, c, h, w);
      for (size_t c = 0; c < y_batch.getShape()[1]; ++c)
        y_batch(j, c) = y_sample(0, c);
    }

    // --- Ciclo de entrenamiento para el batch ---
    // 1. Forward pass
    Tensor logits = model.forward(X_batch, true);
    total_loss += loss_fn.calculate(logits, y_batch);
    total_accuracy += calculate_accuracy(logits, y_batch);

    // 2. Backward pass
    Tensor grad = loss_fn.backward(logits, y_batch);
    model.backward(grad);

    // 3. Actualizacion de parametros
    auto params = model.getParameters();
    auto grads = model.getGradients();
    optimizer.update(params, grads);

    std::cout << "\rEntrenando... Batch " << i + 1 << "/" << num_batches << " " << std::flush;
  }

  return {total_loss / num_batches, total_accuracy / num_batches};
}

// Evalua el rendimiento del modelo, calculando perdida y precision.
std::pair<float, float> Trainer::evaluate(const Tensor &X_test, const Tensor &y_test) {
  size_t num_test_samples = X_test.getShape()[0];
  size_t num_batches = (num_test_samples + config.batch_size - 1) / config.batch_size;

  float total_loss = 0.0f;
  float total_accuracy = 0.0f;

  for (size_t i = 0; i < num_batches; ++i) {
    size_t start = i * config.batch_size;
    size_t count = std::min(config.batch_size, num_test_samples - start);
    if (count == 0)
      continue;

    Tensor X_batch = X_test.slice(0, start, count);
    Tensor y_batch = y_test.slice(0, start, count);

    // Forward pass en modo inferencia (isTraining = false).
    Tensor logits = model.forward(X_batch, false);

    // Calcular perdida y precision para el batch.
    total_loss += loss_fn.calculate(logits, y_batch);
    total_accuracy += calculate_accuracy(logits, y_batch);
  }

  return {total_loss / num_batches, total_accuracy / num_batches};
}
