#include "layers/Pooling2D.hpp"

#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Constructor de la capa Pooling2D.
 */
Pooling2D::Pooling2D(size_t poolSize, PoolType type, size_t stride) : type(type), poolSize(poolSize), stride(stride) {
  if (poolSize == 0) {
    throw std::invalid_argument("El tamaño del pool no puede ser cero.");
  }

  // Si el stride es 0 (valor por defecto), se asume un stride igual al tamaño del pool.
  // Esto corresponde al caso más común de ventanas de pooling no solapadas.
  if (this->stride == 0) {
    this->stride = this->poolSize;
  }
}

/**
 * @brief Devuelve el nombre de la capa, que depende de su tipo.
 */
std::string Pooling2D::getName() const { return (this->type == PoolType::Max) ? "MaxPooling" : "AveragePooling"; }

/**
 * @brief Realiza el paso hacia adelante de la operación de pooling.
 */
Tensor Pooling2D::forward(const Tensor &input, bool isTraining) {
  if (isTraining) {
    this->inputShape = input.getShape();
  }

  const auto &inShape = input.getShape();
  const size_t batchSize = inShape[0];
  const size_t channels = inShape[1];
  const size_t inH = inShape[2];
  const size_t inW = inShape[3];

  // Calcular las dimensiones de la salida
  const size_t outH = (inH - poolSize) / stride + 1;
  const size_t outW = (inW - poolSize) / stride + 1;

  Tensor output({batchSize, channels, outH, outW});

  // Si es Max Pooling y estamos entrenando, inicializamos el tensor para guardar los índices.
  if (this->type == PoolType::Max && isTraining) {
    this->maxIndices = Tensor({batchSize, channels, outH, outW});
  }

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batchSize; ++b) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t oh = 0; oh < outH; ++oh) {
        for (size_t ow = 0; ow < outW; ++ow) {
          // Coordenadas de la esquina superior izquierda de la ventana en la entrada
          size_t h_start = oh * stride;
          size_t w_start = ow * stride;

          float result_val = 0.0f;

          if (this->type == PoolType::Max) {
            result_val = -std::numeric_limits<float>::infinity();
            size_t max_h_idx = 0, max_w_idx = 0;

            // Encontrar el máximo en la ventana
            for (size_t ph = 0; ph < poolSize; ++ph) {
              for (size_t pw = 0; pw < poolSize; ++pw) {
                float current_val = input(b, c, h_start + ph, w_start + pw);
                if (current_val > result_val) {
                  result_val = current_val;
                  max_h_idx = h_start + ph;
                  max_w_idx = w_start + pw;
                }
              }
            }
            // Si estamos entrenando, guardar la posición del máximo
            if (isTraining) {
              // Se guarda el índice plano del máximo (respecto a la imagen completa)
              // para una recuperación eficiente en el backward pass.
              maxIndices(b, c, oh, ow) = static_cast<float>(max_h_idx * inW + max_w_idx);
            }
          } else { // Average Pooling
            float sum = 0.0f;
            for (size_t ph = 0; ph < poolSize; ++ph) {
              for (size_t pw = 0; pw < poolSize; ++pw) {
                sum += input(b, c, h_start + ph, w_start + pw);
              }
            }
            result_val = sum / (poolSize * poolSize);
          }
          output(b, c, oh, ow) = result_val;
        }
      }
    }
  }
  return output;
}

/**
 * @brief Realiza el paso hacia atrás, enrutando o distribuyendo el gradiente.
 */
Tensor Pooling2D::backward(const Tensor &outputGradient) {
  Tensor inputGradient(this->inputShape); // Se inicializa a ceros

  const auto &outGradShape = outputGradient.getShape();
  const size_t batchSize = outGradShape[0];
  const size_t channels = outGradShape[1];
  const size_t outH = outGradShape[2];
  const size_t outW = outGradShape[3];
  const size_t inW = this->inputShape[3]; // Necesario para decodificar el índice plano

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batchSize; ++b) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t oh = 0; oh < outH; ++oh) {
        for (size_t ow = 0; ow < outW; ++ow) {
          float grad = outputGradient(b, c, oh, ow);

          if (this->type == PoolType::Max) {
            // El gradiente solo fluye hacia la posición que fue el máximo.
            size_t flat_idx = static_cast<size_t>(this->maxIndices(b, c, oh, ow));
            size_t max_h = flat_idx / inW;
            size_t max_w = flat_idx % inW;

            // Usamos suma atómica por si hay solapamiento de ventanas (stride < poolSize)
#pragma omp atomic
            inputGradient(b, c, max_h, max_w) += grad;

          } else { // Average Pooling
            // El gradiente se distribuye uniformemente entre todas las neuronas de la ventana.
            size_t h_start = oh * stride;
            size_t w_start = ow * stride;
            float avg_grad = grad / (poolSize * poolSize);

            for (size_t ph = 0; ph < poolSize; ++ph) {
              for (size_t pw = 0; pw < poolSize; ++pw) {
                // Usamos suma atómica aquí también por seguridad con el solapamiento.
#pragma omp atomic
                inputGradient(b, c, h_start + ph, w_start + pw) += avg_grad;
              }
            }
          }
        }
      }
    }
  }
  return inputGradient;
}
