#include "core/Tensor.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// --- Implementación de Métodos Privados ---

/**
 * @brief Calcula los strides para un tensor row-major.
 * @details El stride de una dimensión indica cuántos elementos hay que saltar
 * en la memoria 1D para moverse un paso en esa dimensión.
 * Ejemplo: para una forma {A, B, C}, los strides son {B*C, C, 1}.
 */
void Tensor::computeStrides() {
  strides.resize(shape.size());
  size_t stride = 1;
  // Se itera desde la última dimensión hacia la primera.
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

// --- Implementación de Constructores ---

/** @brief Constructor por defecto: crea un tensor nulo. */
Tensor::Tensor() : dataOffset(0), totalSize(0) {}

/**
 * @brief Constructor de un "Owning Tensor" (propietario).
 * @details Crea la memoria para los datos y la inicializa a cero.
 */
Tensor::Tensor(const std::vector<size_t> &newShape) : shape(newShape), dataOffset(0) {
  totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  dataPtr = std::make_shared<std::vector<float>>(totalSize, 0.0f);
  computeStrides();
}

/**
 * @brief Constructor de un "Owning Tensor" con datos iniciales.
 */
Tensor::Tensor(const std::vector<size_t> &newShape, const std::vector<float> &initialData) : shape(newShape), dataOffset(0) {
  totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  if (totalSize != initialData.size()) {
    throw std::invalid_argument("El tamaño de los datos iniciales no coincide con la forma del tensor.");
  }
  dataPtr = std::make_shared<std::vector<float>>(initialData);
  computeStrides();
}

/**
 * @brief Constructor privado para crear vistas (slices).
 * @details Reutiliza el puntero de datos y los strides del tensor original.
 */
Tensor::Tensor(std::shared_ptr<std::vector<float>> ptr, const std::vector<size_t> &newShape,
               const std::vector<size_t> &originalStrides, size_t offset)
    : dataPtr(ptr), shape(newShape), strides(originalStrides), dataOffset(offset) {
  totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

// --- Implementación de Operaciones y Vistas ---

/**
 * @brief Crea una vista (slice) de bajo coste a lo largo de la primera dimensión.
 */
Tensor Tensor::slice(size_t start, size_t count) const {
  if (shape.empty()) {
    throw std::runtime_error("No se puede hacer slice de un tensor vacío.");
  }
  if (start + count > shape[0]) {
    throw std::out_of_range("Slice fuera de los límites de la primera dimensión.");
  }

  std::vector<size_t> newShape = shape;
  newShape[0] = count;
  // El nuevo offset es el offset actual más el desplazamiento del slice.
  size_t newOffset = dataOffset + start * strides[0];

  // Llama al constructor privado para crear la vista.
  return Tensor(this->dataPtr, newShape, this->strides, newOffset);
}

/**
 * @brief Devuelve la transpuesta de una matriz (tensor 2D).
 */
Tensor Tensor::transpose() const {
  if (shape.size() != 2) {
    throw std::runtime_error("Transpose solo implementado para tensores 2D.");
  }
  Tensor result({shape[1], shape[0]});
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      result(j, i) = (*this)(i, j); // Usa los operadores () para acceso seguro a strides/offsets.
    }
  }
  return result;
}

/**
 * @brief Devuelve un nuevo tensor con el cuadrado de cada elemento.
 */
Tensor Tensor::square() const {
  Tensor result(this->shape); // Crea un tensor del mismo tamaño.
  const auto &current_shape = this->getShape();

  if (current_shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < current_shape[0]; ++i) {
      for (size_t j = 0; j < current_shape[1]; ++j) {
        float val = (*this)(i, j);
        result(i, j) = val * val;
      }
    }
  } else if (current_shape.size() == 4) {
#pragma omp parallel for collapse(4)
    for (size_t b = 0; b < current_shape[0]; ++b) {
      for (size_t c = 0; c < current_shape[1]; ++c) {
        for (size_t h = 0; h < current_shape[2]; ++h) {
          for (size_t w = 0; w < current_shape[3]; ++w) {
            float val = (*this)(b, c, h, w);
            result(b, c, h, w) = val * val;
          }
        }
      }
    }
  } else {
    // Implementación genérica para cualquier número de dimensiones (menos optimizada)
    float *resultData = result.getData();
    const float *thisData = this->getData();
    // NOTA: Esta implementación genérica solo funciona para tensores no-vistas (dataOffset=0)
    // Se deja para demostrar una posible fallback, pero la especialización es mejor.
    if (this->dataOffset != 0) {
      throw std::runtime_error("Tensor::square() para N-dims no soporta vistas.");
    }
#pragma omp parallel for
    for (size_t i = 0; i < totalSize; ++i) {
      resultData[i] = thisData[i] * thisData[i];
    }
  }
  return result;
}

/**
 * @brief Suma los elementos de un tensor a lo largo de un eje.
 * @details Reduce la dimensión del eje a 1, acumulando los valores.
 *          La implementación está especializada para tensores 2D y 4D.
 */
Tensor Tensor::sum(size_t axis) const {
  if (axis >= shape.size()) {
    throw std::out_of_range("Axis fuera de rango para la operación de suma.");
  }

  // La forma de salida tiene la misma cantidad de dimensiones,
  // pero la dimensión 'axis' se colapsa a 1.
  std::vector<size_t> outputShape = this->shape;
  outputShape[axis] = 1;

  Tensor result(outputShape); // Se inicializa a ceros por defecto.

  // --- Caso para tensores 4D (ej. imágenes de Conv2D) ---
  if (shape.size() == 4) {
    // Para cada elemento en el tensor de SALIDA, calculamos su suma.
#pragma omp parallel for collapse(4)
    for (size_t b = 0; b < outputShape[0]; ++b) {
      for (size_t c = 0; c < outputShape[1]; ++c) {
        for (size_t h = 0; h < outputShape[2]; ++h) {
          for (size_t w = 0; w < outputShape[3]; ++w) {
            float current_sum = 0.0f;
            // Iteramos sobre la dimensión que estamos colapsando
            for (size_t i = 0; i < this->shape[axis]; ++i) {
              // Construimos el índice de acceso para el tensor de ENTRADA
              std::vector<size_t> input_idx = {b, c, h, w};
              input_idx[axis] = i; // Sobrescribimos el índice del eje a sumar
              current_sum += (*this)(input_idx[0], input_idx[1], input_idx[2], input_idx[3]);
            }
            result(b, c, h, w) = current_sum;
          }
        }
      }
    }
  }
  // --- Caso para tensores 2D (ej. matrices de Dense) ---
  else if (shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < outputShape[0]; ++i) {
      for (size_t j = 0; j < outputShape[1]; ++j) {
        float current_sum = 0.0f;
        for (size_t k = 0; k < this->shape[axis]; ++k) {
          std::vector<size_t> input_idx = {i, j};
          input_idx[axis] = k;
          current_sum += (*this)(input_idx[0], input_idx[1]);
        }
        result(i, j) = current_sum;
      }
    }
  }
  // --- Caso para cualquier otra dimensión (no implementado aún) ---
  else {
    throw std::runtime_error("Suma por eje (sum) solo implementada para tensores 2D y 4D.");
  }

  return result;
}

/**
 * @brief Suma un vector fila (tensor de forma {1, N}) a cada fila de este tensor.
 * @details Esto es una operación de "broadcasting".
 */
void Tensor::addBroadcast(const Tensor &other) {
  if (shape.size() != 2 || other.shape.size() != 2) {
    throw std::runtime_error("addBroadcast solo implementado para tensores 2D.");
  }
  // Se espera que 'this' sea MxN y 'other' sea 1xN (un vector de bias).
  if (shape[1] != other.shape[1] || other.shape[0] != 1) {
    throw std::runtime_error("Error de broadcast: las dimensiones no coinciden. Se esperaba " + shapeToString() + " y {1, " +
                             std::to_string(shape[1]) + "}.");
  }

#pragma omp parallel for
  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      (*this)(i, j) += other(0, j);
    }
  }
}

// --- Implementación de Getters y Utilidades ---

/** @brief Devuelve un puntero de escritura. Lanza excepción si es una vista compleja. */
float *Tensor::getData() {
  if (dataOffset != 0 || totalSize != dataPtr->size()) {
    // Advertencia: getData() en una vista puede ser ambiguo. El puntero apunta al inicio
    // del bloque de memoria COMPLETO, no al inicio de la vista. Se permite pero con cuidado.
  }
  return dataPtr->data();
}

/** @brief Devuelve un puntero de solo lectura. */
const float *Tensor::getData() const { return dataPtr->data(); }

/** @brief Rellena el tensor con un valor (solo para "Owning Tensors"). */
void Tensor::fill(float value) {
  if (dataOffset != 0 || totalSize != dataPtr->size()) {
    throw std::runtime_error("fill() solo se puede usar en tensores dueños, no en vistas complejas.");
  }
#pragma omp parallel for
  for (size_t i = 0; i < totalSize; ++i) {
    (*dataPtr)[i] = value;
  }
}

/** @brief Rellena con valores aleatorios (solo para "Owning Tensors"). */
void Tensor::randomize(float min, float max) {
  if (dataOffset != 0 || totalSize != dataPtr->size()) {
    throw std::runtime_error("randomize() solo se puede usar en tensores dueños, no en vistas complejas.");
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);

  // std::generate es una buena opción para rellenar contenedores.
  std::generate(dataPtr->begin(), dataPtr->end(), [&]() { return dis(gen); });
}

/** @brief Convierte la forma del tensor a un string legible. */
std::string Tensor::shapeToString() const {
  if (shape.empty())
    return "()";
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    ss << shape[i] << (i == shape.size() - 1 ? "" : ", ");
  }
  ss << ")";
  return ss.str();
}

// --- Implementación de Funciones Libres ---

/**
 * @brief Multiplicación de matrices (GEMM: General Matrix Multiply).
 * @details Multiplica una matriz A (m x n) por una matriz B (n x p), resultando en C (m x p).
 */
Tensor matrixMultiply(const Tensor &a, const Tensor &b) {
  const auto &aShape = a.getShape();
  const auto &bShape = b.getShape();

  if (aShape.size() != 2 || bShape.size() != 2) {
    throw std::runtime_error("La multiplicacion de matrices solo está implementada para tensores 2D.");
  }
  if (aShape[1] != bShape[0]) {
    throw std::runtime_error("Dimensiones de matriz incompatibles para la multiplicacion: " + a.shapeToString() + " y " +
                             b.shapeToString());
  }

  const size_t m = aShape[0];
  const size_t n = aShape[1];
  const size_t p = bShape[1];

  Tensor result({m, p}); // Tensor dueño para el resultado.

// Se paraleliza el bucle más externo para distribuir el trabajo por filas del resultado.
#pragma omp parallel for
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < n; ++k) {
        // El uso de a(i, k) y b(k, j) asegura que se manejen correctamente los
        // strides y offsets si 'a' o 'b' fueran vistas (slices).
        sum += a(i, k) * b(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}
