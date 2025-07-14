#include "layers/Dense.hpp"
#include "core/Tensor.hpp"
#include <stdexcept>

Dense::Dense(size_t inputSize, size_t outputSize) {
  // Inicializacion de pesos con He.
  float stddev = std::sqrt(2.0f / static_cast<float>(inputSize));
  this->weights = Tensor({inputSize, outputSize});
  this->weights.randomizeNormal(0.0f, stddev);

  // Inicializacion de bias a cero. Forma {1, outputSize} para broadcasting.
  this->bias = Tensor({1, outputSize});
  this->bias.fill(0.0f);

  // Inicializacion de gradientes con la misma forma, a cero.
  this->weightGradients = Tensor({inputSize, outputSize});
  this->biasGradients = Tensor({1, outputSize});
}

Tensor Dense::forward(const Tensor &input, bool isTraining) {
  if (isTraining) {
    // Guarda la entrada para el calculo en backward.
    this->inputTensor = input;
  }

  const auto &inputShape = input.getShape();
  size_t inputRank = inputShape.size();

  // Caso 3D: {batch, tokens, features_in} -> {batch, tokens, features_out}
  if (inputRank == 3) {
    size_t batchSize = inputShape[0];
    size_t numTokens = inputShape[1];
    size_t featuresIn = inputShape[2];
    // Aplana a 2D para la multiplicacion.
    Tensor input2D = input.reshape({batchSize * numTokens, featuresIn});

    Tensor output2D = matrixMultiply(input2D, this->weights);
    output2D.addBroadcast(this->bias);

    // Devuelve la forma original 3D.
    return output2D.reshape({batchSize, numTokens, this->bias.getShape()[1]});
  }

  // Caso 2D: {batch, features_in} -> {batch, features_out}
  if (inputRank == 2) {
    Tensor output = matrixMultiply(input, this->weights);
    output.addBroadcast(this->bias);
    return output;
  }

  throw std::runtime_error("Dense::forward solo soporta entradas 2D o 3D.");
}

Tensor Dense::backward(const Tensor &outputGradient) {
  const auto &inputShape = this->inputTensor.getShape();
  size_t inputRank = inputShape.size();

  Tensor grad_to_process = outputGradient;
  Tensor input_to_process = this->inputTensor;

  // Si la entrada original era 3D, se aplana el gradiente y la entrada guardada.
  if (inputRank == 3) {
    size_t batchSize = inputShape[0];
    size_t numTokens = inputShape[1];
    size_t featuresIn = inputShape[2];
    size_t featuresOut = outputGradient.getShape()[2];

    // Asegura que los tensores son contiguos antes de aplanar.
    if (!grad_to_process.isContiguous()) {
      grad_to_process = grad_to_process.contiguous();
    }
    if (!input_to_process.isContiguous()) {
      input_to_process = input_to_process.contiguous();
    }

    grad_to_process = grad_to_process.reshape({batchSize * numTokens, featuresOut});
    input_to_process = input_to_process.reshape({batchSize * numTokens, featuresIn});
  }

  // Calculos de gradientes (siempre se hacen en 2D).
  // dE/dW = X^T * dE/dY
  Tensor inputTransposed = input_to_process.transpose(0, 1);
  this->weightGradients = matrixMultiply(inputTransposed, grad_to_process);

  // dE/db = sum(dE/dY) a lo largo del eje del batch.
  this->biasGradients = grad_to_process.sum(0);

  // dE/dX = dE/dY * W^T
  Tensor weightsTransposed = this->weights.transpose(0, 1);
  Tensor inputGradient2D = matrixMultiply(grad_to_process, weightsTransposed);

  // Si la entrada original era 3D, se devuelve el gradiente a su forma 3D.
  if (inputRank == 3) {
    return inputGradient2D.reshape(inputShape);
  }

  return inputGradient2D;
}

std::vector<Tensor *> Dense::getParameters() { return {&this->weights, &this->bias}; }

std::vector<Tensor *> Dense::getGradients() { return {&this->weightGradients, &this->biasGradients}; }
