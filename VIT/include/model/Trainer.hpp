#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "losses/CrossEntropy.hpp"
#include "model/VisionTransformer.hpp"
#include "optimizers/Adam.hpp"
#include <memory>
#include <vector>

// Estructura para los hiperparametros del entrenamiento.
struct TrainerConfig {
  int epochs = 10;
  size_t batch_size = 64;
  float learning_rate = 0.001f;
  float weight_decay = 0.01f;
};

// Clase que orquesta el proceso de entrenamiento del modelo.
class Trainer {
public:
  // Constructor. Recibe el modelo y la configuracion de entrenamiento.
  Trainer(VisionTransformer &model, const TrainerConfig &train_config);

  // Ejecuta el bucle de entrenamiento completo.
  // - train_data: Par {Imagenes, Etiquetas} para el entrenamiento.
  // - test_data: Par {Imagenes, Etiquetas} para la validacion.
  void train(const std::pair<Tensor, Tensor> &train_data, const std::pair<Tensor, Tensor> &test_data);

  // Getters para acceder al modelo.
  const VisionTransformer &getModel() const { return model; }
  VisionTransformer &getModel() { return model; }

private:
  // Ejecuta una unica epoca de entrenamiento sobre el conjunto de datos.
  // Devuelve la perdida y precision promedio de la epoca.
  std::pair<float, float> train_epoch(const Tensor &X_train, const Tensor &y_train);

  // Evalua el modelo en un conjunto de datos (sin actualizar pesos).
  // Devuelve la perdida y precision promedio.
  std::pair<float, float> evaluate(const Tensor &X_test, const Tensor &y_test);

  // Componentes del entrenamiento.
  VisionTransformer &model; // Referencia al modelo a entrenar.
  Adam optimizer;
  CrossEntropy loss_fn;

  // Configuracion de entrenamiento.
  TrainerConfig config;
};

#endif // TRAINER_HPP
