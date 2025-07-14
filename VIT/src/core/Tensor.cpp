#include "core/Tensor.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// --- Implementacion de Metodos Privados ---

// Calcula los strides para un tensor contiguo (row-major).
// El stride de una dimension indica cuantos elementos saltar en memoria para
// moverse un paso en esa dimension.
// Ejemplo: forma {A, B, C}, strides {B*C, C, 1}.
void Tensor::computeStrides() {
  strides.resize(shape.size());
  if (shape.empty())
    return;

  size_t stride = 1;
  // Itera desde la ultima dimension hacia la primera.
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

// --- Implementacion de Constructores ---

// Constructor por defecto: crea un tensor nulo.
Tensor::Tensor() : dataOffset(0), totalSize(0) {}

// Constructor de un tensor "propietario" (dueño de la memoria).
// Crea la memoria para los datos y la inicializa a cero.
Tensor::Tensor(const std::vector<size_t> &newShape) : shape(newShape), dataOffset(0) {
  totalSize = newShape.empty() ? 0 : std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<size_t>());
  dataPtr = std::make_shared<std::vector<float>>(totalSize, 0.0f);
  computeStrides();
}

// Constructor de un tensor "propietario" con datos iniciales.
Tensor::Tensor(const std::vector<size_t> &newShape, const std::vector<float> &initialData) : shape(newShape), dataOffset(0) {
  totalSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<size_t>());
  if (totalSize != initialData.size()) {
    throw std::invalid_argument("El tamaño de los datos iniciales no coincide con la forma del tensor.");
  }
  dataPtr = std::make_shared<std::vector<float>>(initialData);
  computeStrides();
}

// Constructor privado para crear vistas (slices, reshapes, etc.).
// Reutiliza el puntero de datos del tensor original.
Tensor::Tensor(std::shared_ptr<std::vector<float>> ptr, const std::vector<size_t> &newShape,
               const std::vector<size_t> &newStrides, size_t offset)
    : dataPtr(std::move(ptr)), shape(newShape), strides(newStrides), dataOffset(offset) {
  totalSize = shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

// --- Getters y Utilidades ---

// Devuelve un puntero de escritura al inicio del bloque de datos.
float *Tensor::getData() {
  if (!dataPtr)
    return nullptr;
  return dataPtr->data();
}

// Devuelve un puntero de solo lectura al inicio del bloque de datos.
const float *Tensor::getData() const {
  if (!dataPtr)
    return nullptr;
  return dataPtr->data();
}

// Comprueba si el tensor es contiguo en memoria.
// Un tensor es contiguo si sus strides siguen el patron row-major.
// Esto es importante para operaciones de bajo nivel como memcpy.
bool Tensor::isContiguous() const {
  if (shape.empty())
    return true;

  size_t stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    if (strides[i] != stride) {
      return false;
    }
    stride *= shape[i];
  }
  return true;
}

// Convierte la forma del tensor a una cadena de texto legible.
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

// --- Inicializacion y Modificacion ---

// Rellena el tensor con un valor escalar. Solo para tensores contiguos.
void Tensor::fill(float value) {
  if (!isContiguous()) {
    throw std::runtime_error("fill() solo se puede usar en tensores contiguos.");
  }
  if (dataPtr) {
    float *start_ptr = this->getData() + this->dataOffset;
    std::fill(start_ptr, start_ptr + totalSize, value);
  }
}

// Rellena con valores aleatorios. Solo para tensores contiguos.
void Tensor::randomize(float min, float max) {
  if (!isContiguous()) {
    throw std::runtime_error("randomize() solo se puede usar en tensores contiguos.");
  }
  if (dataPtr) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    float *start_ptr = this->getData() + this->dataOffset;
    std::generate(start_ptr, start_ptr + totalSize, [&]() { return dis(gen); });
  }
}

// Rellena con valores de una distribucion normal. Solo para tensores contiguos.
void Tensor::randomizeNormal(float mean, float stddev) {
  if (!isContiguous()) {
    throw std::runtime_error("randomizeNormal() solo se puede usar en tensores contiguos.");
  }
  if (dataPtr) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, stddev);

    float *start_ptr = this->getData() + this->dataOffset;
    std::generate(start_ptr, start_ptr + totalSize, [&]() { return dis(gen); });
  }
}

// --- Operaciones de Creacion de Vistas ---

// Crea una vista (slice) del tensor a lo largo de un eje.
// No copia datos, solo ajusta la forma, el offset y reutiliza los strides.
Tensor Tensor::slice(size_t axis, size_t start, size_t count) const {
  if (axis >= shape.size()) {
    throw std::out_of_range("Eje de slice fuera de rango.");
  }
  if (start + count > shape[axis]) {
    throw std::out_of_range("Slice fuera de los limites de la dimension " + std::to_string(axis));
  }

  std::vector<size_t> newShape = shape;
  newShape[axis] = count;

  // El nuevo offset se calcula a partir del stride del eje especificado.
  size_t newOffset = dataOffset + start * strides[axis];

  // Se reutilizan los strides originales, la disposicion relativa no cambia.
  return Tensor(this->dataPtr, newShape, this->strides, newOffset);
}

// Reinterpreta la forma del tensor sin copiar datos.
// Solo funciona si el tensor es contiguo.
Tensor Tensor::reshape(const std::vector<size_t> &newShape) const {
  if (!isContiguous()) {
    throw std::runtime_error("reshape() solo se puede usar en un tensor contiguo. Use .contiguous() primero.");
  }
  size_t newTotalSize = newShape.empty() ? 0 : std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<size_t>());
  if (this->totalSize != newTotalSize) {
    throw std::runtime_error("No se puede hacer reshape: el numero total de elementos debe ser el mismo.");
  }
  // Crea un tensor temporal solo para calcular los nuevos strides.
  Tensor tempForStrides(newShape);
  return Tensor(this->dataPtr, newShape, tempForStrides.getStrides(), this->dataOffset);
}

// Devuelve una vista transpuesta del tensor intercambiando dos dimensiones.
// No copia datos, solo intercambia los valores de shape y strides.
Tensor Tensor::transpose(size_t dim1, size_t dim2) const {
  if (dim1 >= shape.size() || dim2 >= shape.size()) {
    throw std::out_of_range("Ejes para transpose fuera de rango.");
  }
  std::vector<size_t> newShape = this->shape;
  std::swap(newShape[dim1], newShape[dim2]);
  std::vector<size_t> newStrides = this->strides;
  std::swap(newStrides[dim1], newStrides[dim2]);
  return Tensor(this->dataPtr, newShape, newStrides, this->dataOffset);
}

// Devuelve una copia contigua en memoria del tensor.
// Si el tensor ya es contiguo, se devuelve a si mismo sin copiar.
Tensor Tensor::contiguous() const {
  // Si ya es contiguo y no es una vista con offset, no hay nada que hacer.
  if (isContiguous() && dataOffset == 0) {
    return *this;
  }

  Tensor new_tensor(this->shape);
  // Usa los operadores () que manejan strides para copiar elemento por elemento
  // desde el tensor (posiblemente no contiguo) al nuevo tensor contiguo.
  if (shape.size() == 4) {
#pragma omp parallel for collapse(4)
    for (size_t d0 = 0; d0 < shape[0]; ++d0)
      for (size_t d1 = 0; d1 < shape[1]; ++d1)
        for (size_t d2 = 0; d2 < shape[2]; ++d2)
          for (size_t d3 = 0; d3 < shape[3]; ++d3)
            new_tensor(d0, d1, d2, d3) = (*this)(d0, d1, d2, d3);
  } else if (shape.size() == 3) {
#pragma omp parallel for collapse(3)
    for (size_t d0 = 0; d0 < shape[0]; ++d0)
      for (size_t d1 = 0; d1 < shape[1]; ++d1)
        for (size_t d2 = 0; d2 < shape[2]; ++d2)
          new_tensor(d0, d1, d2) = (*this)(d0, d1, d2);
  } else {
    throw std::runtime_error("contiguous() no implementado para este rank.");
  }

  return new_tensor;
}

// --- Operaciones Aritmeticas ---

// Suma dos tensores elemento por elemento. Deben tener la misma forma.
Tensor Tensor::operator+(const Tensor &other) const {
  if (this->shape != other.getShape()) {
    throw std::invalid_argument("Los tensores deben tener la misma forma para la suma. " + this->shapeToString() + " vs " +
                                other.shapeToString());
  }

  Tensor result(this->shape);

  // Iteramos sobre el tensor de salida. El uso de los operadores () asegura
  // que funcione correctamente incluso para tensores no contiguos (vistas).
  if (this->shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < this->shape[0]; ++i) {
      for (size_t j = 0; j < this->shape[1]; ++j) {
        result(i, j) = (*this)(i, j) + other(i, j);
      }
    }
  } else if (this->shape.size() == 3) {
#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < this->shape[0]; ++i) {
      for (size_t j = 0; j < this->shape[1]; ++j) {
        for (size_t k = 0; k < this->shape[2]; ++k) {
          result(i, j, k) = (*this)(i, j, k) + other(i, j, k);
        }
      }
    }
  } else { // Fallback para 1D u otras formas (solo funciona si es contiguo).
    float *result_data = result.getData();
    const float *this_data = this->getData();
    const float *other_data = other.getData();
    for (size_t i = 0; i < this->totalSize; ++i) {
      result_data[i] = this_data[dataOffset + i] + other_data[other.getDataOffset() + i];
    }
  }

  return result;
}

// Devuelve un nuevo tensor con el cuadrado de cada elemento.
Tensor Tensor::square() const {
  Tensor result(this->shape);

  if (this->shape.size() == 2) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < this->shape[0]; ++i) {
      for (size_t j = 0; j < this->shape[1]; ++j) {
        result(i, j) = (*this)(i, j) * (*this)(i, j);
      }
    }
  } else if (this->shape.size() == 3) {
#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < this->shape[0]; ++i) {
      for (size_t j = 0; j < this->shape[1]; ++j) {
        for (size_t k = 0; k < this->shape[2]; ++k) {
          result(i, j, k) = (*this)(i, j, k) * (*this)(i, j, k);
        }
      }
    }
  } else { // Fallback (solo funciona si es contiguo).
    float *result_data = result.getData();
    const float *this_data = this->getData();
    for (size_t i = 0; i < this->totalSize; ++i) {
      result_data[i] = this_data[dataOffset + i] * this_data[dataOffset + i];
    }
  }
  return result;
}

// Suma los elementos de un tensor a lo largo de un eje, reduciendo su tamaño a 1.
Tensor Tensor::sum(size_t axis) const {
  if (axis >= shape.size()) {
    throw std::out_of_range("Eje para sum() fuera de rango.");
  }

  std::vector<size_t> outputShape = this->shape;
  outputShape[axis] = 1;
  Tensor result(outputShape); // Se inicializa a ceros.

  // Se itera sobre la forma de salida y se suma a lo largo del eje colapsado de la entrada.
  // Esto es mas lento pero funciona para cualquier dimensionalidad y vista.
  if (shape.size() == 2) {
#pragma omp parallel for
    for (size_t i = 0; i < outputShape[0]; ++i) {
      for (size_t j = 0; j < outputShape[1]; ++j) {
        float current_sum = 0.0f;
        // Itera sobre la dimension que se esta sumando.
        for (size_t k = 0; k < this->shape[axis]; ++k) {
          std::vector<size_t> idx = {i, j};
          idx[axis] = k;
          current_sum += (*this)(idx[0], idx[1]);
        }
        result(i, j) = current_sum;
      }
    }
  } else if (shape.size() == 3) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < outputShape[0]; ++i) {
      for (size_t j = 0; j < outputShape[1]; ++j) {
        for (size_t k = 0; k < outputShape[2]; ++k) {
          float current_sum = 0.0f;
          // Itera sobre la dimension que se esta sumando.
          for (size_t l = 0; l < this->shape[axis]; ++l) {
            std::vector<size_t> idx = {i, j, k};
            idx[axis] = l;
            current_sum += (*this)(idx[0], idx[1], idx[2]);
          }
          result(i, j, k) = current_sum;
        }
      }
    }
  } else {
    throw std::runtime_error("sum() solo esta implementado para 2D y 3D por ahora.");
  }

  return result;
}

// Suma un tensor 'other' a este, aplicando broadcasting.
void Tensor::addBroadcast(const Tensor &other) {
  // Caso 1: Broadcasting de {1, N} sobre {M, N}
  if (this->shape.size() == 2 && other.getShape().size() == 2 && other.getShape()[0] == 1 &&
      this->shape[1] == other.getShape()[1]) {
#pragma omp parallel for
    for (size_t i = 0; i < this->shape[0]; ++i) {
      for (size_t j = 0; j < this->shape[1]; ++j) {
        (*this)(i, j) += other(0, j);
      }
    }
  }
  // Caso 2: Broadcasting de {1, N, D} sobre {B, N, D}
  else if (this->shape.size() == 3 && other.getShape().size() == 3 && other.getShape()[0] == 1 &&
           this->shape[1] == other.getShape()[1] && this->shape[2] == other.getShape()[2]) {
#pragma omp parallel for collapse(2)
    for (size_t b = 0; b < this->shape[0]; ++b) {
      for (size_t n = 0; n < this->shape[1]; ++n) {
        for (size_t d = 0; d < this->shape[2]; ++d) {
          (*this)(b, n, d) += other(0, n, d);
        }
      }
    }
  } else {
    throw std::runtime_error("Broadcasting no implementado para estas formas: " + this->shapeToString() + " y " +
                             other.shapeToString());
  }
}

// Realiza la multiplicacion de matrices (GEMM: General Matrix Multiply).
// Multiplica una matriz A (m x n) por una matriz B (n x p), resultando en C (m x p).
// Funciona correctamente con vistas (slices, transposiciones) gracias al acceso `()`.
Tensor matrixMultiply(const Tensor &a, const Tensor &b) {
  const auto &aShape = a.getShape();
  const auto &bShape = b.getShape();

  if (aShape.size() != 2 || bShape.size() != 2) {
    throw std::runtime_error("matrixMultiply solo esta implementada para tensores 2D.");
  }
  if (aShape[1] != bShape[0]) {
    throw std::runtime_error("Dimensiones de matriz incompatibles para la multiplicacion: " + a.shapeToString() + " y " +
                             b.shapeToString());
  }

  const size_t m = aShape[0];
  const size_t n = aShape[1];
  const size_t p = bShape[1];

  Tensor result({m, p});

  // Se paraleliza el bucle mas externo.
#pragma omp parallel for
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < n; ++k) {
        // El uso de a(i, k) y b(k, j) asegura que se manejen correctamente los
        // strides y offsets si 'a' o 'b' son vistas.
        sum += a(i, k) * b(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}

// Realiza la multiplicacion de matrices por lotes (BMM: Batched Matrix Multiply).
// Multiplica un tensor A (B x m x n) por un tensor B (B x n x p), resultando C (B x m x p).
Tensor batchMatrixMultiply(const Tensor &a, const Tensor &b) {
  const auto &aShape = a.getShape();
  const auto &bShape = b.getShape();

  if (aShape.size() != 3 || bShape.size() != 3) {
    throw std::runtime_error("BMM solo esta implementado para tensores 3D.");
  }
  if (aShape[0] != bShape[0]) {
    throw std::runtime_error("El tamaño del batch debe ser el mismo para ambos tensores en BMM.");
  }
  if (aShape[2] != bShape[1]) {
    throw std::runtime_error("Dimensiones de matriz incompatibles para BMM: " + a.shapeToString() + " y " + b.shapeToString());
  }

  const size_t batchSize = aShape[0];
  const size_t m = aShape[1];
  const size_t n = aShape[2];
  const size_t p = bShape[2];

  Tensor result({batchSize, m, p});

#pragma omp parallel for
  for (size_t i = 0; i < batchSize; ++i) {
    for (size_t j = 0; j < m; ++j) {
      for (size_t k = 0; k < p; ++k) {
        float sum = 0.0f;
        for (size_t l = 0; l < n; ++l) {
          sum += a(i, j, l) * b(i, l, k);
        }
        result(i, j, k) = sum;
      }
    }
  }
  return result;
}

// Concatena una lista de tensores a lo largo de un eje especifico.
// Todos los tensores deben tener las mismas dimensiones excepto en el eje de concatenacion.
Tensor concatenate(const std::vector<Tensor> &tensors, size_t axis) {
  if (tensors.empty()) {
    return Tensor();
  }
  if (tensors.size() == 1) {
    return tensors[0];
  }

  // 1. Validaciones
  const auto &firstShape = tensors[0].getShape();
  size_t newDimSize = 0;
  for (const auto &t : tensors) {
    if (t.getShape().size() != firstShape.size() || t.getShape().size() <= axis) {
      throw std::invalid_argument("Todos los tensores deben tener el mismo rank y ser compatibles con el eje.");
    }
    for (size_t i = 0; i < firstShape.size(); ++i) {
      if (i != axis && t.getShape()[i] != firstShape[i]) {
        throw std::invalid_argument("Las dimensiones deben ser iguales excepto en el eje de concatenacion.");
      }
    }
    newDimSize += t.getShape()[axis];
  }

  // 2. Calcular la nueva forma y crear el tensor resultado
  std::vector<size_t> newShape = firstShape;
  newShape[axis] = newDimSize;
  Tensor result(newShape);

  // 3. Copiar los datos de cada tensor en la seccion correcta del resultado
  size_t offset_on_axis = 0;
  for (const auto &t : tensors) {
    // Crea una vista (slice) en el tensor de resultado donde se copiaran los datos.
    Tensor result_slice = result.slice(axis, offset_on_axis, t.getShape()[axis]);

    // Copia los datos respetando los strides.
    if (t.getShape().size() == 3) { // Especializado para 3D
#pragma omp parallel for collapse(3)
      for (size_t i = 0; i < t.getShape()[0]; ++i) {
        for (size_t j = 0; j < t.getShape()[1]; ++j) {
          for (size_t k = 0; k < t.getShape()[2]; ++k) {
            result_slice(i, j, k) = t(i, j, k);
          }
        }
      }
    } else {
      throw std::runtime_error("Concatenate solo implementado para 3D por ahora.");
    }

    offset_on_axis += t.getShape()[axis];
  }

  return result;
}

// Crea una vista de un tensor con una dimension extra de tamaño 'size'.
// No copia datos. Lo logra estableciendo el stride de la nueva dimension a 0.
Tensor expand(const Tensor &tensor, size_t dim, size_t size) {
  if (dim > tensor.getShape().size()) {
    throw std::invalid_argument("La dimension para expandir es mayor que el rank del tensor.");
  }

  std::vector<size_t> newShape = tensor.getShape();
  newShape.insert(newShape.begin() + dim, size);

  std::vector<size_t> newStrides = tensor.getStrides();
  // El stride para la nueva dimension es 0.
  // Al acceder, cualquier indice en esta dimension se multiplica por 0,
  // reutilizando los mismos datos.
  newStrides.insert(newStrides.begin() + dim, 0);

  return Tensor(tensor.getDataPtr(), newShape, newStrides, tensor.getDataOffset());
}

// Imprime informacion de depuracion sobre el estado interno del tensor.
void Tensor::printDebugInfo(const std::string &name) const {
  std::cout << "--- Tensor Debug: " << name << " ---" << std::endl;
  std::cout << "  Forma: " << shapeToString() << std::endl;
  std::cout << "  Contiguo: " << (isContiguous() ? "Si" : "NO") << std::endl;
  std::cout << "  Offset: " << dataOffset << std::endl;
  std::cout << "  Strides: ";
  for (const auto &s : strides)
    std::cout << s << " ";
  std::cout << std::endl;
  std::cout << "-------------------------" << std::endl;
}
