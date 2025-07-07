#include "layers/Conv2D.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

// Incluir OpenMP si está disponible
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Constructor de la capa Conv2D.
 */
Conv2D::Conv2D(size_t inChannels, size_t outChannels, size_t kernelSize, size_t stride, size_t padding)
    : inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize), stride(stride), padding(padding) {
  // Inicialización de pesos usando una variante de la inicialización de Glorot/Xavier.
  // fan_in y fan_out ayudan a escalar los pesos para mantener la varianza de la señal.
  float fan_in = static_cast<float>(inChannels * kernelSize * kernelSize);
  float fan_out = static_cast<float>(outChannels * kernelSize * kernelSize);
  float limit = std::sqrt(6.0f / (fan_in + fan_out)); // Fórmula de Glorot

  // Forma de los pesos: {num_filtros, canales_entrada, alto_kernel, ancho_kernel}
  this->weights = Tensor({outChannels, inChannels, kernelSize, kernelSize});
  this->weights.randomize(-limit, limit);

  // Un bias por cada filtro/canal de salida.
  this->bias = Tensor({1, outChannels});
  this->bias.fill(0.0f);

  // Inicializar gradientes con las mismas formas, rellenos de ceros.
  this->weightGradients = Tensor(this->weights.getShape());
  this->biasGradients = Tensor(this->bias.getShape());
}

/**
 * @brief Realiza el paso hacia adelante de la convolución usando la técnica im2col.
 */
Tensor Conv2D::forward(const Tensor &input, bool isTraining) {
  if (isTraining) {
    this->inputShape = input.getShape();
  }
  const size_t batchSize = input.getShape()[0];
  const size_t inH = input.getShape()[2];
  const size_t inW = input.getShape()[3];

  // 1. Calcular dimensiones de salida
  const size_t outH = (inH + 2 * padding - kernelSize) / stride + 1;
  const size_t outW = (inW + 2 * padding - kernelSize) / stride + 1;

  // 2. Transformar la entrada a una matriz de columnas (im2col)
  // Esta matriz se guarda como miembro (this->im2colMatrix) para reutilizarla en el backward pass.
  this->im2col(input, outH, outW);

  // 3. Remodelar los pesos de los filtros para la multiplicación de matrices.
  // De {outC, inC, kH, kW} a {outC, inC*kH*kW}
  Tensor reshapedWeights({this->outChannels, this->inChannels * this->kernelSize * this->kernelSize});
  std::copy(this->weights.getData(), this->weights.getData() + this->weights.getSize(), reshapedWeights.getData());

  // 4. Realizar la convolución como una única multiplicación de matrices.
  // Resultado: {outC, B*outH*outW}
  Tensor convResult = matrixMultiply(reshapedWeights, this->im2colMatrix);

  // 5. Remodelar la salida y añadir el bias.
  Tensor output({batchSize, this->outChannels, outH, outW});
#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batchSize; ++b) {
    for (size_t oc = 0; oc < this->outChannels; ++oc) {
      for (size_t i = 0; i < outH * outW; ++i) {
        size_t oh = i / outW;
        size_t ow = i % outW;
        size_t col_idx = b * (outH * outW) + i; // Índice correspondiente en la matriz de resultado
        output(b, oc, oh, ow) = convResult(oc, col_idx) + this->bias(0, oc);
      }
    }
  }
  return output;
}

/**
 * @brief Realiza el paso hacia atrás de la convolución.
 */
Tensor Conv2D::backward(const Tensor &outputGradient) {
  const size_t batchSize = outputGradient.getShape()[0];
  const size_t outH = outputGradient.getShape()[2];
  const size_t outW = outputGradient.getShape()[3];

  // --- 1. Calcular el gradiente del bias (dE/db) ---
  // El gradiente de cada bias es la suma de los gradientes de salida de su mapa de características.
  this->biasGradients = outputGradient.sum(0).sum(2).sum(3); // Suma sobre B, H, W

  // --- 2. Calcular el gradiente de los pesos (dE/dW) ---
  // dE/dW = dE/dY * (im2colMatrix)^T
  Tensor reshapedOutGrad({this->outChannels, batchSize * outH * outW});
  // TODO: Evitar esta copia si es posible. Un `Tensor::reshape` sin copia sería ideal.
  std::copy(outputGradient.getData(), outputGradient.getData() + outputGradient.getSize(), reshapedOutGrad.getData());

  Tensor im2colTransposed = this->im2colMatrix.transpose();
  Tensor flatWeightGradients = matrixMultiply(reshapedOutGrad, im2colTransposed);

  // TODO: Evitar esta copia. `weightGradients` podría construirse como una vista de `flatWeightGradients`.
  std::copy(flatWeightGradients.getData(), flatWeightGradients.getData() + flatWeightGradients.getSize(),
            this->weightGradients.getData());

  // --- 3. Calcular el gradiente de la entrada (dE/dX) ---
  // dE/dX se obtiene mediante la "convolución transpuesta" del gradiente de salida con los pesos.
  // En el paradigma im2col, esto es: dL/dX_col = W^T * dL/dY
  Tensor reshapedWeights({this->outChannels, this->inChannels * this->kernelSize * this->kernelSize});
  std::copy(this->weights.getData(), this->weights.getData() + this->weights.getSize(), reshapedWeights.getData());

  Tensor transposedWeights = reshapedWeights.transpose();
  Tensor dL_dX_col = matrixMultiply(transposedWeights, reshapedOutGrad);

  // Ahora, transformamos la matriz de gradientes de columna de vuelta a una "imagen".
  Tensor inputGradient(this->inputShape);
  this->col2im(dL_dX_col, inputGradient);

  return inputGradient;
}

// --- Métodos de utilidad im2col y col2im ---

/**
 * @brief Transforma parches de la imagen de entrada en columnas de una matriz.
 * @details Cada columna de la matriz de salida representa un parche de la imagen de entrada
 *          aplanado en un vector.
 */
void Conv2D::im2col(const Tensor &input, size_t outH, size_t outW) {
  const size_t batchSize = input.getShape()[0];
  const size_t inH = input.getShape()[2];
  const size_t inW = input.getShape()[3];
  const size_t colRows = this->inChannels * this->kernelSize * this->kernelSize;
  const size_t colCols = batchSize * outH * outW;
  this->im2colMatrix = Tensor({colRows, colCols});

#pragma omp parallel for
  for (size_t col_idx = 0; col_idx < colCols; ++col_idx) {
    // Calcular a qué parche (b, oh, ow) corresponde esta columna
    size_t b = col_idx / (outH * outW);
    size_t oh = (col_idx / outW) % outH;
    size_t ow = col_idx % outW;

    size_t row_idx = 0;
    // Rellenar la columna iterando sobre el parche correspondiente
    for (size_t ic = 0; ic < this->inChannels; ++ic) {
      for (size_t kh = 0; kh < this->kernelSize; ++kh) {
        for (size_t kw = 0; kw < this->kernelSize; ++kw) {
          int h_in = static_cast<int>(oh * this->stride + kh) - static_cast<int>(this->padding);
          int w_in = static_cast<int>(ow * this->stride + kw) - static_cast<int>(this->padding);

          float val = 0.0f;
          if (h_in >= 0 && h_in < static_cast<int>(inH) && w_in >= 0 && w_in < static_cast<int>(inW)) {
            val = input(b, ic, h_in, w_in);
          }
          this->im2colMatrix(row_idx, col_idx) = val;
          row_idx++;
        }
      }
    }
  }
}

/**
 * @brief Operación inversa a im2col. Transforma una matriz de columnas en una "imagen".
 * @details Acumula los valores de las columnas en las posiciones correctas de un tensor de
 *          salida. Esencial para calcular el gradiente de entrada (dE/dX).
 */
void Conv2D::col2im(const Tensor &colMatrix, Tensor &outputImage) {
  const size_t batchSize = outputImage.getShape()[0];
  const size_t imgH = outputImage.getShape()[2]; // Altura de la imagen de salida (sin padding)
  const size_t imgW = outputImage.getShape()[3]; // Anchura de la imagen de salida (sin padding)
  const size_t outH = (imgH + 2 * padding - kernelSize) / stride + 1;
  const size_t outW = (imgW + 2 * padding - kernelSize) / stride + 1;
  outputImage.fill(0.0f); // Es crucial empezar con ceros porque vamos a acumular.

#pragma omp parallel for
  for (size_t col_idx = 0; col_idx < colMatrix.getShape()[1]; ++col_idx) {
    // Calcular a qué parche (b, oh, ow) corresponde esta columna
    size_t b = col_idx / (outH * outW);
    size_t oh = (col_idx / outW) % outH;
    size_t ow = col_idx % outW;

    size_t row_idx = 0;
    for (size_t ic = 0; ic < this->inChannels; ++ic) {
      for (size_t kh = 0; kh < this->kernelSize; ++kh) {
        for (size_t kw = 0; kw < this->kernelSize; ++kw) {
          int h_img = static_cast<int>(oh * this->stride + kh) - static_cast<int>(this->padding);
          int w_img = static_cast<int>(ow * this->stride + kw) - static_cast<int>(this->padding);

          if (h_img >= 0 && h_img < static_cast<int>(imgH) && w_img >= 0 && w_img < static_cast<int>(imgW)) {
            // Un píxel de la imagen de entrada puede ser parte de varios parches,
            // por lo que sus gradientes deben sumarse.
#pragma omp atomic
            outputImage(b, ic, h_img, w_img) += colMatrix(row_idx, col_idx);
          }
          row_idx++;
        }
      }
    }
  }
}

// --- Getters ---

std::vector<Tensor *> Conv2D::getParameters() { return {&this->weights, &this->bias}; }

std::vector<Tensor *> Conv2D::getGradients() { return {&this->weightGradients, &this->biasGradients}; }
