#include "layers/Dense.hpp"
#include "core/Tensor.hpp" // Para matrixMultiply y otras operaciones de Tensor
#include <stdexcept>

/**
 * @brief Constructor de la capa Dense.
 * @details Inicializa los tensores de pesos, bias y sus correspondientes gradientes.
 *          Los pesos se inicializan usando la inicialización de Glorot/Xavier para
 *          ayudar a prevenir problemas de gradientes que se desvanecen o explotan.
 */
Dense::Dense(size_t inputSize, size_t outputSize) {
  // Inicialización de pesos usando una variante de la inicialización de Glorot/Xavier.
  // Un buen esquema de inicialización es crucial para la convergencia del entrenamiento.
  float limit = std::sqrt(6.0f / (inputSize + outputSize));
  // Tu versión original (sqrt(6.0f / inputSize)) también es una variante válida (LeCun init).
  // La de Glorot es más común. Dejamos la de Glorot por ser un estándar.

  this->weights = Tensor({inputSize, outputSize});
  this->weights.randomize(-limit, limit);

  // El bias se inicializa a cero.
  this->bias = Tensor({1, outputSize});
  this->bias.fill(0.0f);

  // Los gradientes se inicializan a cero con las mismas formas que los parámetros.
  this->weightGradients = Tensor({inputSize, outputSize});
  this->biasGradients = Tensor({1, outputSize});
}

/**
 * @brief Realiza el paso hacia adelante: Y = X * W + b.
 */
Tensor Dense::forward(const Tensor &input, bool isTraining) {
  // Si estamos entrenando, es crucial guardar la entrada.
  // La necesitaremos en el backward pass para calcular dE/dW.
  if (isTraining) {
    this->inputTensor = input; // inputTensor es una copia.
  }

  // 1. Multiplicación de la matriz de entrada por los pesos: Y' = X * W
  Tensor output = matrixMultiply(input, this->weights);

  // 2. Añadir el bias usando broadcasting: Y = Y' + b
  output.addBroadcast(this->bias);

  return output;
}

/**
 * @brief Realiza la retropropagación a través de la capa.
 */
Tensor Dense::backward(const Tensor &outputGradient) {
  // --- Cálculo de gradientes de los parámetros ---

  // 1. Gradiente de los pesos (dE/dW):
  //    La derivada de (X * W) respecto a W es X^T.
  //    Aplicando la regla de la cadena: dE/dW = dE/dY * (dY/dW) = dE/dY * X^T
  //    En notación matricial, esto es: X^T * dE/dY
  Tensor inputTransposed = this->inputTensor.transpose();
  this->weightGradients = matrixMultiply(inputTransposed, outputGradient);

  // 2. Gradiente del bias (dE/db):
  //    La derivada de (Y + b) respecto a b es 1.
  //    Aplicando la regla de la cadena: dE/db = dE/dY * (dY/db) = dE/dY
  //    Como el bias se suma a cada muestra del batch, su gradiente es la suma
  //    de los gradientes de salida a lo largo de la dimensión del batch (axis=0).
  this->biasGradients = outputGradient.sum(0);

  // --- Cálculo del gradiente para la capa anterior ---

  // 3. Gradiente de la entrada (dE/dX):
  //    La derivada de (X * W) respecto a X es W^T.
  //    Aplicando la regla de la cadena: dE/dX = dE/dY * (dY/dX) = dE/dY * W^T
  //    Este es el gradiente que se propaga hacia atrás.
  Tensor weightsTransposed = this->weights.transpose();
  Tensor inputGradient = matrixMultiply(outputGradient, weightsTransposed);

  return inputGradient;
}

/**
 * @brief Proporciona acceso a los parámetros entrenables.
 */
std::vector<Tensor *> Dense::getParameters() { return {&this->weights, &this->bias}; }

/**
 * @brief Proporciona acceso a los gradientes calculados.
 * @details El orden DEBE coincidir con getParameters().
 */
std::vector<Tensor *> Dense::getGradients() { return {&this->weightGradients, &this->biasGradients}; }
