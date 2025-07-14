#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// --- Declaraciones adelantadas de funciones ---
class Tensor; // Declaracion adelantada de la clase Tensor.

// Concatena una lista de tensores a lo largo de un eje especifico.
Tensor concatenate(const std::vector<Tensor> &tensors, size_t axis);

// Expande un tensor replicando sus datos a lo largo de una nueva dimension.
Tensor expand(const Tensor &tensor, size_t dim, size_t size);

// Representa un tensor N-dimensional, la estructura de datos fundamental.
// Gestiona un bloque de memoria multidimensional de forma eficiente usando
// punteros compartidos, permitiendo vistas (slices, transposes) sin copia de datos.
class Tensor {
public:
  // --- Constructores y Destructor ---

  // Constructor por defecto. Crea un tensor vacio.
  Tensor();
  // Constructor que crea un tensor con la forma especificada, inicializado a cero.
  explicit Tensor(const std::vector<size_t> &shape);
  // Constructor que crea un tensor a partir de una forma y datos existentes.
  Tensor(const std::vector<size_t> &shape, const std::vector<float> &data);
  // Constructores y operadores de copia y movimiento por defecto.
  Tensor(const Tensor &other) = default;
  Tensor(Tensor &&other) noexcept = default;
  Tensor &operator=(const Tensor &other) = default;
  Tensor &operator=(Tensor &&other) noexcept = default;
  // Destructor por defecto.
  ~Tensor() = default;

  // --- Acceso a Elementos (Optimizados para 1D, 2D, 3D y 4D) ---
  float &operator()(size_t i);
  const float &operator()(size_t i) const;
  float &operator()(size_t i, size_t j);
  const float &operator()(size_t i, size_t j) const;
  float &operator()(size_t i, size_t j, size_t k);
  const float &operator()(size_t i, size_t j, size_t k) const;
  float &operator()(size_t d0, size_t d1, size_t d2, size_t d3);
  const float &operator()(size_t d0, size_t d1, size_t d2, size_t d3) const;

  // --- Operaciones y Vistas ---

  // Crea una vista (slice) del tensor a lo largo de un eje. No copia datos.
  Tensor slice(size_t axis, size_t start, size_t count) const;
  // Cambia la forma del tensor. No copia datos si es posible.
  Tensor reshape(const std::vector<size_t> &newShape) const;
  // Intercambia dos dimensiones del tensor. No copia datos.
  Tensor transpose(size_t dim1, size_t dim2) const;
  // Devuelve un nuevo tensor con el cuadrado de cada elemento.
  Tensor square() const;
  // Suma los elementos del tensor a lo largo de un eje especificado.
  Tensor sum(size_t axis) const;
  // Suma otro tensor a este, usando broadcasting si las formas no coinciden.
  void addBroadcast(const Tensor &other);
  // Devuelve una version contigua en memoria de este tensor. Crea una copia si no lo es.
  Tensor contiguous() const;

  // --- Operadores Aritmeticos ---

  // Suma elemento a elemento dos tensores.
  Tensor operator+(const Tensor &other) const;

  // --- Inicializacion y Modificacion ---

  // Rellena todo el tensor con un valor escalar.
  void fill(float value);
  // Inicializa el tensor con valores aleatorios de una distribucion uniforme.
  void randomize(float min = -1.0f, float max = 1.0f);
  // Inicializa el tensor con valores aleatorios de una distribucion normal.
  void randomizeNormal(float mean = 0.0f, float stddev = 1.0f);

  // --- Getters y Utilidades ---

  // Devuelve la forma (dimensiones) del tensor.
  const std::vector<size_t> &getShape() const { return shape; }
  // Devuelve el numero total de elementos en el tensor.
  size_t getSize() const { return totalSize; }
  // Devuelve los strides del tensor.
  const std::vector<size_t> &getStrides() const { return strides; }
  // Devuelve el desplazamiento inicial dentro del bloque de datos.
  size_t getDataOffset() const { return dataOffset; }
  // Devuelve el puntero compartido al vector de datos subyacente.
  const std::shared_ptr<std::vector<float>> &getDataPtr() const { return dataPtr; }
  // Devuelve un puntero constante a los datos brutos del tensor.
  const float *getData() const;
  // Devuelve un puntero a los datos brutos del tensor.
  float *getData();
  // Convierte la forma del tensor a una cadena de texto para impresion.
  std::string shapeToString() const;
  // Verifica si el tensor esta almacenado de forma contigua en memoria.
  bool isContiguous() const;
  // Imprime informacion de depuracion sobre el estado interno del tensor.
  void printDebugInfo(const std::string &name) const;

  Tensor(std::shared_ptr<std::vector<float>> dataPtr, const std::vector<size_t> &shape, const std::vector<size_t> &strides,
         size_t offset);

private:
  // Constructor privado para uso interno, crea vistas eficientes.

  // Calcula los strides a partir de la forma del tensor para acceso multidimensional.
  void computeStrides();

  // Puntero compartido al bloque de datos. Permite que varias vistas compartan memoria.
  std::shared_ptr<std::vector<float>> dataPtr;
  // Dimensiones del tensor (ej. {lote, canales, alto, ancho}).
  std::vector<size_t> shape;
  // Pasos en memoria para navegar cada dimension. Clave para las vistas.
  std::vector<size_t> strides;
  // Desplazamiento inicial en dataPtr. Util para slices.
  size_t dataOffset;
  // Numero total de elementos en el tensor.
  size_t totalSize;
};

// --- Funciones Libres para Operaciones de Tensor ---

// Realiza la multiplicacion de matrices entre dos tensores 2D.
Tensor matrixMultiply(const Tensor &a, const Tensor &b);

// Realiza la multiplicacion de matrices por lotes (BMM) en tensores 3D.
Tensor batchMatrixMultiply(const Tensor &a, const Tensor &b);

// --- Implementaciones Inline (para rendimiento) ---

inline float &Tensor::operator()(size_t i) {
#ifndef NDEBUG
  if (shape.size() != 1 || i >= shape[0])
    throw std::out_of_range("Acceso 1D fuera de rango.");
#endif
  return (*dataPtr)[dataOffset + i * strides[0]];
}

inline const float &Tensor::operator()(size_t i) const {
#ifndef NDEBUG
  if (shape.size() != 1 || i >= shape[0])
    throw std::out_of_range("Acceso 1D fuera de rango.");
#endif
  return (*dataPtr)[dataOffset + i * strides[0]];
}

inline float &Tensor::operator()(size_t i, size_t j) {
#ifndef NDEBUG
  if (shape.size() != 2 || i >= shape[0] || j >= shape[1])
    throw std::out_of_range("Acceso 2D fuera de rango.");
#endif
  return (*dataPtr)[dataOffset + i * strides[0] + j * strides[1]];
}

inline const float &Tensor::operator()(size_t i, size_t j) const {
#ifndef NDEBUG
  if (shape.size() != 2 || i >= shape[0] || j >= shape[1])
    throw std::out_of_range("Acceso 2D fuera de rango.");
#endif
  return (*dataPtr)[dataOffset + i * strides[0] + j * strides[1]];
}

inline float &Tensor::operator()(size_t i, size_t j, size_t k) {
#ifndef NDEBUG
  if (shape.size() != 3 || i >= shape[0] || j >= shape[1] || k >= shape[2])
    throw std::out_of_range("Acceso 3D fuera de rango.");
#endif
  return (*dataPtr)[dataOffset + i * strides[0] + j * strides[1] + k * strides[2]];
}

inline const float &Tensor::operator()(size_t i, size_t j, size_t k) const {
#ifndef NDEBUG
  if (shape.size() != 3 || i >= shape[0] || j >= shape[1] || k >= shape[2])
    throw std::out_of_range("Acceso 3D fuera de rango.");
#endif
  return (*dataPtr)[dataOffset + i * strides[0] + j * strides[1] + k * strides[2]];
}

inline float &Tensor::operator()(size_t d0, size_t d1, size_t d2, size_t d3) {
#ifndef NDEBUG
  if (shape.size() != 4 || d0 >= shape[0] || d1 >= shape[1] || d2 >= shape[2] || d3 >= shape[3])
    throw std::out_of_range("Acceso 4D fuera de rango.");
#endif
  return (*dataPtr)[dataOffset + d0 * strides[0] + d1 * strides[1] + d2 * strides[2] + d3 * strides[3]];
}

inline const float &Tensor::operator()(size_t d0, size_t d1, size_t d2, size_t d3) const {
#ifndef NDEBUG
  if (shape.size() != 4 || d0 >= shape[0] || d1 >= shape[1] || d2 >= shape[2] || d3 >= shape[3])
    throw std::out_of_range("Acceso 4D fuera de rango.");
#endif
  return (*dataPtr)[dataOffset + d0 * strides[0] + d1 * strides[1] + d2 * strides[2] + d3 * strides[3]];
}

#endif // TENSOR_HPP

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

#ifndef LAYER_HPP
#define LAYER_HPP

#include "core/Tensor.hpp"
#include <string>
#include <vector>

// Clase base abstracta para todas las capas de la red.
// Define la interfaz comun para el forward, backward y gestion de parametros.
class Layer {
public:
  // Destructor virtual para herencia polimorfica.
  virtual ~Layer() = default;

  // Realiza el paso hacia adelante (forward pass) de la capa.
  virtual Tensor forward(const Tensor &input, bool isTraining) = 0;

  // Retropropaga el gradiente y calcula los gradientes de los parametros.
  virtual Tensor backward(const Tensor &outputGradient) = 0;

  // Devuelve los parametros entrenables de la capa (pesos, biases).
  virtual std::vector<Tensor *> getParameters() { return {}; }

  // Devuelve los gradientes asociados a los parametros entrenables.
  virtual std::vector<Tensor *> getGradients() { return {}; }

  // Devuelve el nombre de la capa (ej. "Dense").
  virtual std::string getName() const = 0;
};

#endif // LAYER_HPP

#ifndef DENSE_HPP
#define DENSE_HPP

#include "layers/Layer.hpp"
#include <cmath>

// Capa totalmente conectada (fully connected).
// Realiza la operacion: output = input * weights + bias.
class Dense : public Layer {
public:
  // Constructor. Define las dimensiones de entrada y salida.
  Dense(size_t inputSize, size_t outputSize);

  // Realiza la transformacion afin: Y = X * W + b.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Calcula los gradientes para los pesos, el bias y la entrada.
  Tensor backward(const Tensor &outputGradient) override;

  // Devuelve los parametros entrenables: pesos y bias.
  std::vector<Tensor *> getParameters() override;

  // Devuelve los gradientes de los parametros.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "Dense"; }

private:
  // Parametros entrenables
  Tensor weights; // Matriz de pesos, forma {input_size, output_size}.
  Tensor bias;    // Vector de bias, forma {1, output_size}.

  // Gradientes de los parametros
  Tensor weightGradients; // Gradiente de los pesos.
  Tensor biasGradients;   // Gradiente del bias.

  // Almacena la entrada del forward pass para el calculo del backward pass.
  Tensor inputTensor;
};

#endif // DENSE_HPP
//

#include "core/Tensor.hpp"
#include "layers/Dense.hpp"
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

#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP

#include "layers/Layer.hpp"

// Implementa la Normalizacion de Capa (Layer Normalization).
// Normaliza las activaciones a lo largo de la dimension de caracteristicas.
// y = gamma * (x - mean) / sqrt(var + epsilon) + beta
class LayerNorm : public Layer {
public:
  // Constructor. Define el tamaño de la dimension a normalizar.
  LayerNorm(size_t featureSize, float epsilon = 1e-5f);

  // Realiza el paso de normalizacion hacia adelante.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Calcula los gradientes para gamma, beta y la entrada.
  Tensor backward(const Tensor &outputGradient) override;

  // Devuelve los parametros entrenables: gamma y beta.
  std::vector<Tensor *> getParameters() override;

  // Devuelve los gradientes de los parametros.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "LayerNorm"; }

private:
  float epsilon;
  size_t featureSize;

  // Parametros entrenables
  Tensor gamma; // Parametro de escala, forma {1, 1, ..., feature_size}.
  Tensor beta;  // Parametro de desplazamiento, forma {1, 1, ..., feature_size}.

  // Gradientes de los parametros
  Tensor gammaGradient;
  Tensor betaGradient;

  // Estado para el backward pass
  Tensor inputTensor;     // Copia de la entrada del forward.
  Tensor mean;            // Media por cada muestra.
  Tensor variance;        // Varianza por cada muestra.
  Tensor normalizedInput; // Entrada normalizada antes de gamma/beta.
};

#endif // LAYERNOWN_HPP

#include "layers/LayerNorm.hpp"
#include <cmath>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

LayerNorm::LayerNorm(size_t featureSize, float epsilon) : featureSize(featureSize), epsilon(epsilon) {
  // Gamma se inicializa a 1 (sin cambio de escala inicial).
  this->gamma = Tensor({1, featureSize});
  this->gamma.fill(1.0f);

  // Beta se inicializa a 0 (sin desplazamiento inicial).
  this->beta = Tensor({1, featureSize});
  this->beta.fill(0.0f);

  // Gradientes se inicializan a cero.
  this->gammaGradient = Tensor({1, featureSize});
  this->betaGradient = Tensor({1, featureSize});
}

Tensor LayerNorm::forward(const Tensor &input, bool isTraining) {
  const auto &inputShape = input.getShape();
  if (inputShape.back() != this->featureSize) {
    throw std::runtime_error("La ultima dimension de entrada no coincide con featureSize.");
  }

  // El "batch" es el producto de todas las dimensiones excepto la ultima.
  size_t batchSize = input.getSize() / this->featureSize;

  // Aplanamos temporalmente la entrada a 2D para simplificar calculos.
  Tensor input2D = input.reshape({batchSize, this->featureSize});

  // En entrenamiento, guardamos valores intermedios para backward.
  if (isTraining) {
    this->inputTensor = input2D;
    this->mean = Tensor({batchSize, 1});
    this->variance = Tensor({batchSize, 1}); // Se reutilizara para guardar inv_stddev.
  }

  Tensor output2D({batchSize, this->featureSize});
  this->normalizedInput = Tensor({batchSize, this->featureSize});

#pragma omp parallel for
  for (size_t i = 0; i < batchSize; ++i) {
    // --- 1. Calcular la media ---
    float current_mean = 0.0f;
    for (size_t j = 0; j < this->featureSize; ++j) {
      current_mean += input2D(i, j);
    }
    current_mean /= this->featureSize;

    // --- 2. Calcular la varianza ---
    float current_variance = 0.0f;
    for (size_t j = 0; j < this->featureSize; ++j) {
      float diff = input2D(i, j) - current_mean;
      current_variance += diff * diff;
    }
    current_variance /= this->featureSize;

    float inv_stddev = 1.0f / std::sqrt(current_variance + this->epsilon);

    if (isTraining) {
      this->mean(i, 0) = current_mean;
      this->variance(i, 0) = inv_stddev; // Guardamos 1/sqrt(var+eps)
    }

    // --- 3. Normalizar, escalar y desplazar ---
    for (size_t j = 0; j < this->featureSize; ++j) {
      float x_hat = (input2D(i, j) - current_mean) * inv_stddev;
      if (isTraining)
        this->normalizedInput(i, j) = x_hat; // Guardamos la entrada normalizada.

      output2D(i, j) = this->gamma(0, j) * x_hat + this->beta(0, j);
    }
  }

  // Devolvemos el tensor a su forma original.
  return output2D.reshape(inputShape);
}

Tensor LayerNorm::backward(const Tensor &outputGradient) {
  const auto &gradShape = outputGradient.getShape();
  size_t batchSize = outputGradient.getSize() / this->featureSize;

  // Aplanamos el gradiente de salida a 2D.
  Tensor grad2D = outputGradient.reshape({batchSize, this->featureSize});

  // Reseteamos los gradientes de los parametros antes de acumular.
  this->gammaGradient.fill(0.0f);
  this->betaGradient.fill(0.0f);
  Tensor inputGradient({batchSize, this->featureSize});

  // El bucle sobre el batch es secuencial para evitar race conditions al acumular
  // los gradientes de gamma y beta, que son compartidos por todo el batch.
  for (size_t i = 0; i < batchSize; ++i) {
    float inv_stddev = this->variance(i, 0); // Reutilizamos el valor guardado.

    float dL_dXhat_sum = 0;
    float dL_dXhat_dot_Xhat_sum = 0;

    // --- 1. Calcular gradientes de gamma, beta y sumas intermedias ---
    // dL/dgamma = sum(dL/dY * X_hat) ; dL/dbeta = sum(dL/dY)
    for (size_t j = 0; j < this->featureSize; ++j) {
      float grad_y_ij = grad2D(i, j);
      float x_hat_ij = this->normalizedInput(i, j);

      this->gammaGradient(0, j) += grad_y_ij * x_hat_ij;
      this->betaGradient(0, j) += grad_y_ij;

      float dL_dXhat = grad_y_ij * this->gamma(0, j);
      dL_dXhat_sum += dL_dXhat;
      dL_dXhat_dot_Xhat_sum += dL_dXhat * x_hat_ij;
    }

    // --- 2. Calcular el gradiente de la entrada (dL/dX) ---
    // Se aplica la formula completa derivada de la normalizacion.
    for (size_t j = 0; j < this->featureSize; ++j) {
      float dL_dXhat_ij = grad2D(i, j) * this->gamma(0, j);
      float x_hat_ij = this->normalizedInput(i, j);

      float term1 = this->featureSize * dL_dXhat_ij;
      float term2 = dL_dXhat_sum;
      float term3 = x_hat_ij * dL_dXhat_dot_Xhat_sum;

      inputGradient(i, j) = (1.0f / this->featureSize) * inv_stddev * (term1 - term2 - term3);
    }
  }

  // Devolvemos el gradiente a su forma original.
  return inputGradient.reshape(gradShape);
}

std::vector<Tensor *> LayerNorm::getParameters() { return {&this->gamma, &this->beta}; }

std::vector<Tensor *> LayerNorm::getGradients() { return {&this->gammaGradient, &this->betaGradient}; }

#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP

#include "layers/Layer.hpp"

// Implementa la Normalizacion de Capa (Layer Normalization).
// Normaliza las activaciones a lo largo de la dimension de caracteristicas.
// y = gamma * (x - mean) / sqrt(var + epsilon) + beta
class LayerNorm : public Layer {
public:
  // Constructor. Define el tamaño de la dimension a normalizar.
  LayerNorm(size_t featureSize, float epsilon = 1e-5f);

  // Realiza el paso de normalizacion hacia adelante.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Calcula los gradientes para gamma, beta y la entrada.
  Tensor backward(const Tensor &outputGradient) override;

  // Devuelve los parametros entrenables: gamma y beta.
  std::vector<Tensor *> getParameters() override;

  // Devuelve los gradientes de los parametros.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "LayerNorm"; }

private:
  float epsilon;
  size_t featureSize;

  // Parametros entrenables
  Tensor gamma; // Parametro de escala, forma {1, 1, ..., feature_size}.
  Tensor beta;  // Parametro de desplazamiento, forma {1, 1, ..., feature_size}.

  // Gradientes de los parametros
  Tensor gammaGradient;
  Tensor betaGradient;

  // Estado para el backward pass
  Tensor inputTensor;     // Copia de la entrada del forward.
  Tensor mean;            // Media por cada muestra.
  Tensor variance;        // Varianza por cada muestra.
  Tensor normalizedInput; // Entrada normalizada antes de gamma/beta.
};

#endif // LAYERNOWN_HPP

#include "layers/LayerNorm.hpp"
#include <cmath>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

LayerNorm::LayerNorm(size_t featureSize, float epsilon) : featureSize(featureSize), epsilon(epsilon) {
  // Gamma se inicializa a 1 (sin cambio de escala inicial).
  this->gamma = Tensor({1, featureSize});
  this->gamma.fill(1.0f);

  // Beta se inicializa a 0 (sin desplazamiento inicial).
  this->beta = Tensor({1, featureSize});
  this->beta.fill(0.0f);

  // Gradientes se inicializan a cero.
  this->gammaGradient = Tensor({1, featureSize});
  this->betaGradient = Tensor({1, featureSize});
}

Tensor LayerNorm::forward(const Tensor &input, bool isTraining) {
  const auto &inputShape = input.getShape();
  if (inputShape.back() != this->featureSize) {
    throw std::runtime_error("La ultima dimension de entrada no coincide con featureSize.");
  }

  // El "batch" es el producto de todas las dimensiones excepto la ultima.
  size_t batchSize = input.getSize() / this->featureSize;

  // Aplanamos temporalmente la entrada a 2D para simplificar calculos.
  Tensor input2D = input.reshape({batchSize, this->featureSize});

  // En entrenamiento, guardamos valores intermedios para backward.
  if (isTraining) {
    this->inputTensor = input2D;
    this->mean = Tensor({batchSize, 1});
    this->variance = Tensor({batchSize, 1}); // Se reutilizara para guardar inv_stddev.
  }

  Tensor output2D({batchSize, this->featureSize});
  this->normalizedInput = Tensor({batchSize, this->featureSize});

#pragma omp parallel for
  for (size_t i = 0; i < batchSize; ++i) {
    // --- 1. Calcular la media ---
    float current_mean = 0.0f;
    for (size_t j = 0; j < this->featureSize; ++j) {
      current_mean += input2D(i, j);
    }
    current_mean /= this->featureSize;

    // --- 2. Calcular la varianza ---
    float current_variance = 0.0f;
    for (size_t j = 0; j < this->featureSize; ++j) {
      float diff = input2D(i, j) - current_mean;
      current_variance += diff * diff;
    }
    current_variance /= this->featureSize;

    float inv_stddev = 1.0f / std::sqrt(current_variance + this->epsilon);

    if (isTraining) {
      this->mean(i, 0) = current_mean;
      this->variance(i, 0) = inv_stddev; // Guardamos 1/sqrt(var+eps)
    }

    // --- 3. Normalizar, escalar y desplazar ---
    for (size_t j = 0; j < this->featureSize; ++j) {
      float x_hat = (input2D(i, j) - current_mean) * inv_stddev;
      if (isTraining)
        this->normalizedInput(i, j) = x_hat; // Guardamos la entrada normalizada.

      output2D(i, j) = this->gamma(0, j) * x_hat + this->beta(0, j);
    }
  }

  // Devolvemos el tensor a su forma original.
  return output2D.reshape(inputShape);
}

Tensor LayerNorm::backward(const Tensor &outputGradient) {
  const auto &gradShape = outputGradient.getShape();
  size_t batchSize = outputGradient.getSize() / this->featureSize;

  // Aplanamos el gradiente de salida a 2D.
  Tensor grad2D = outputGradient.reshape({batchSize, this->featureSize});

  // Reseteamos los gradientes de los parametros antes de acumular.
  this->gammaGradient.fill(0.0f);
  this->betaGradient.fill(0.0f);
  Tensor inputGradient({batchSize, this->featureSize});

  // El bucle sobre el batch es secuencial para evitar race conditions al acumular
  // los gradientes de gamma y beta, que son compartidos por todo el batch.
  for (size_t i = 0; i < batchSize; ++i) {
    float inv_stddev = this->variance(i, 0); // Reutilizamos el valor guardado.

    float dL_dXhat_sum = 0;
    float dL_dXhat_dot_Xhat_sum = 0;

    // --- 1. Calcular gradientes de gamma, beta y sumas intermedias ---
    // dL/dgamma = sum(dL/dY * X_hat) ; dL/dbeta = sum(dL/dY)
    for (size_t j = 0; j < this->featureSize; ++j) {
      float grad_y_ij = grad2D(i, j);
      float x_hat_ij = this->normalizedInput(i, j);

      this->gammaGradient(0, j) += grad_y_ij * x_hat_ij;
      this->betaGradient(0, j) += grad_y_ij;

      float dL_dXhat = grad_y_ij * this->gamma(0, j);
      dL_dXhat_sum += dL_dXhat;
      dL_dXhat_dot_Xhat_sum += dL_dXhat * x_hat_ij;
    }

    // --- 2. Calcular el gradiente de la entrada (dL/dX) ---
    // Se aplica la formula completa derivada de la normalizacion.
    for (size_t j = 0; j < this->featureSize; ++j) {
      float dL_dXhat_ij = grad2D(i, j) * this->gamma(0, j);
      float x_hat_ij = this->normalizedInput(i, j);

      float term1 = this->featureSize * dL_dXhat_ij;
      float term2 = dL_dXhat_sum;
      float term3 = x_hat_ij * dL_dXhat_dot_Xhat_sum;

      inputGradient(i, j) = (1.0f / this->featureSize) * inv_stddev * (term1 - term2 - term3);
    }
  }

  // Devolvemos el gradiente a su forma original.
  return inputGradient.reshape(gradShape);
}

std::vector<Tensor *> LayerNorm::getParameters() { return {&this->gamma, &this->beta}; }

std::vector<Tensor *> LayerNorm::getGradients() { return {&this->gammaGradient, &this->betaGradient}; }

#ifndef FEEDFORWARD_HPP
#define FEEDFORWARD_HPP

#include "activations/GELU.hpp"
#include "layers/Dense.hpp"
#include "layers/Layer.hpp"
#include <vector>

// Implementa la red Feed-Forward (MLP) del bloque Transformer.
// Consiste en dos capas lineales con una activacion no lineal en medio:
// Dense -> GELU -> Dense.
class FeedForward : public Layer {
public:
  // Constructor. Define la dimension de entrada/salida y la dimension oculta.
  FeedForward(size_t embedding_dim, size_t hidden_dim);

  // Realiza el paso hacia adelante a traves de las capas internas.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Realiza el paso hacia atras en orden inverso al forward.
  Tensor backward(const Tensor &outputGradient) override;

  // Recolecta los parametros de las capas Dense internas.
  std::vector<Tensor *> getParameters() override;

  // Recolecta los gradientes de las capas Dense internas.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "FeedForward"; }

private:
  // Capas que componen la red Feed-Forward.
  Dense dense1;
  GELU activation;
  Dense dense2;
};

#endif // FEEDFORWARD_HPP
//

#include "layers/FeedForward.hpp"

// Constructor que inicializa las sub-capas en la lista de inicializadores.
FeedForward::FeedForward(size_t embedding_dim, size_t hidden_dim)
    : dense1(embedding_dim, hidden_dim),              // Capa 1: entrada -> oculta
      activation(), dense2(hidden_dim, embedding_dim) // Capa 2: oculta -> salida
{}

// Encadena el forward pass de las sub-capas: dense1 -> activation -> dense2.
Tensor FeedForward::forward(const Tensor &input, bool isTraining) {
  Tensor x = dense1.forward(input, isTraining);
  x = activation.forward(x, isTraining);
  x = dense2.forward(x, isTraining);
  return x;
}

// Encadena el backward pass de las sub-capas en orden inverso.
Tensor FeedForward::backward(const Tensor &outputGradient) {
  Tensor grad = dense2.backward(outputGradient);
  grad = activation.backward(grad);
  grad = dense1.backward(grad);
  return grad;
}

// Recolecta los parametros de las dos capas Dense.
std::vector<Tensor *> FeedForward::getParameters() {
  auto params1 = dense1.getParameters();
  auto params2 = dense2.getParameters();
  // Concatena los vectores de parametros.
  params1.insert(params1.end(), params2.begin(), params2.end());
  return params1;
}

// Recolecta los gradientes de las dos capas Dense.
std::vector<Tensor *> FeedForward::getGradients() {
  auto grads1 = dense1.getGradients();
  auto grads2 = dense2.getGradients();
  // Concatena los vectores de gradientes.
  grads1.insert(grads1.end(), grads2.begin(), grads2.end());
  return grads1;
}

#ifndef MULTIHEADATTENTION_HPP
#define MULTIHEADATTENTION_HPP

#include "layers/Dense.hpp"
#include "layers/Layer.hpp"
#include <memory>
#include <vector>

// Implementa el mecanismo de Atencion Multi-Cabeza (Multi-Head Attention).
// Es el componente central de los bloques Transformer.
class MultiHeadAttention : public Layer {
public:
  // Constructor.
  // - embedding_dim: Dimension de los embeddings de entrada y salida (D).
  // - num_heads: Numero de cabezas de atencion (h). Debe dividir a D.
  MultiHeadAttention(size_t embedding_dim, size_t num_heads);

  // Realiza el paso hacia adelante de la atencion.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Realiza el paso hacia atras.
  Tensor backward(const Tensor &outputGradient) override;

  // Recolecta los parametros de las capas Dense internas (Q, K, V, Out).
  std::vector<Tensor *> getParameters() override;

  // Recolecta los gradientes de las capas Dense internas.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "MultiHeadAttention"; }

private:
  size_t embedding_dim;
  size_t num_heads;
  size_t head_dim; // Dimension de cada cabeza (D / h).

  // Capas de proyeccion lineal para Query, Key, Value y la salida.
  std::unique_ptr<Dense> q_proj;
  std::unique_ptr<Dense> k_proj;
  std::unique_ptr<Dense> v_proj;
  std::unique_ptr<Dense> out_proj;

  // Funcion auxiliar para la atencion escalada por producto punto.
  Tensor scaledDotProductAttention(const Tensor &q, const Tensor &k, const Tensor &v);

  // Tensores guardados para el backward pass.
  Tensor inputTensor;               // Entrada original.
  Tensor q_split, k_split, v_split; // Proyecciones Q, K, V divididas por cabeza.
  Tensor attention_weights;         // Pesos de atencion despues de softmax.
};

#endif // MULTIHEADATTENTION_HPP

#include "core/Tensor.hpp"
#include "layers/MultiHeadAttention.hpp"
#include <cmath>
#include <limits>

// --- Declaraciones de funciones auxiliares ---
Tensor softmax(const Tensor &logits, int axis);
Tensor softmax_backward(const Tensor &grad_output, const Tensor &softmax_output);

MultiHeadAttention::MultiHeadAttention(size_t embedding_dim, size_t num_heads)
    : embedding_dim(embedding_dim), num_heads(num_heads) {

  if (embedding_dim % num_heads != 0) {
    throw std::invalid_argument("embedding_dim debe ser divisible por num_heads.");
  }
  this->head_dim = embedding_dim / num_heads;

  // Inicializa las cuatro proyecciones lineales.
  q_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
  k_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
  v_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
  out_proj = std::make_unique<Dense>(embedding_dim, embedding_dim);
}

Tensor MultiHeadAttention::forward(const Tensor &input, bool isTraining) {
  if (isTraining) {
    this->inputTensor = input;
  }

  const auto &s = input.getShape(); // {B, N, D}
  size_t B = s[0], N = s[1];

  // 1. Proyecciones Lineales para obtener Q, K, V.
  Tensor q = q_proj->forward(input, isTraining); // -> {B, N, D}
  Tensor k = k_proj->forward(input, isTraining); // -> {B, N, D}
  Tensor v = v_proj->forward(input, isTraining); // -> {B, N, D}

  // 2. Dividir Q, K, V en cabezas de atencion.
  // {B, N, D} -> {B, N, h, d_h} -> {B, h, N, d_h}
  auto split_heads = [&](Tensor t) {
    t = t.reshape({B, N, this->num_heads, this->head_dim});
    t = t.transpose(1, 2); // Ahora es no-contiguo.
    t = t.contiguous();    // Lo hacemos contiguo para el proximo reshape.
    return t.reshape({B * this->num_heads, N, this->head_dim});
  };
  q = split_heads(q);
  k = split_heads(k);
  v = split_heads(v);

  if (isTraining) {
    this->q_split = q;
    this->k_split = k;
    this->v_split = v;
  }

  // 3. Atencion Escalada por Producto Punto.
  Tensor context = scaledDotProductAttention(q, k, v); // -> {B*h, N, d_h}

  // 4. Re-ensamblar las cabezas.
  // {B*h, N, d_h} -> {B, h, N, d_h} -> {B, N, h, d_h}
  context = context.reshape({B, this->num_heads, N, this->head_dim});
  context = context.transpose(1, 2); // Crea una vista no-contigua.
  context = context.contiguous();    // La hacemos contigua para el reshape final.
  context = context.reshape({B, N, this->embedding_dim});

  // 5. Proyeccion de salida final.
  return out_proj->forward(context, isTraining);
}

Tensor MultiHeadAttention::scaledDotProductAttention(const Tensor &q, const Tensor &k, const Tensor &v) {
  // scores = (Q * K^T) / sqrt(d_k)
  Tensor k_transposed = k.transpose(1, 2);              // -> {B*h, d_h, N}
  Tensor scores = batchMatrixMultiply(q, k_transposed); // -> {B*h, N, N}

  float scale_factor = 1.0f / std::sqrt(static_cast<float>(this->head_dim));

  // Escala los scores.
  if (scores.isContiguous()) {
    float *scores_data = scores.getData();
#pragma omp parallel for
    for (size_t i = 0; i < scores.getSize(); ++i) {
      scores_data[i] *= scale_factor;
    }
  } else { // Fallback para vistas no contiguas.
    const auto &s = scores.getShape();
    for (size_t i = 0; i < s[0]; ++i)
      for (size_t j = 0; j < s[1]; ++j)
        for (size_t l = 0; l < s[2]; ++l)
          scores(i, j, l) *= scale_factor;
  }

  // Aplica softmax para obtener los pesos de atencion.
  Tensor attention = softmax(scores, 2);
  if (this->attention_weights.getShape() != attention.getShape()) {
    this->attention_weights = Tensor(attention.getShape());
  }
  this->attention_weights = attention;

  // context = attention_weights * V
  return batchMatrixMultiply(attention, v);
}

Tensor MultiHeadAttention::backward(const Tensor &outputGradient) {
  const auto &inputShape = this->inputTensor.getShape();
  size_t B = inputShape[0], N = inputShape[1];

  // 1. Invertir proyeccion de salida.
  Tensor grad_context = this->out_proj->backward(outputGradient); // dL/d(context), forma {B, N, D}

  // 2. Invertir re-ensamblaje de cabezas.
  grad_context = grad_context.reshape({B, N, this->num_heads, this->head_dim});
  grad_context = grad_context.transpose(1, 2).contiguous();
  grad_context = grad_context.reshape({B * this->num_heads, N, this->head_dim});
  // grad_context es ahora dL/d(attention_output)

  // 3. Invertir attention_output = attention_weights @ V
  Tensor V_T = this->v_split.transpose(1, 2);
  Tensor d_attention_weights = batchMatrixMultiply(grad_context, V_T);

  Tensor attention_weights_T = this->attention_weights.transpose(1, 2);
  Tensor dV = batchMatrixMultiply(attention_weights_T, grad_context);

  // 4. Invertir Softmax.
  Tensor d_scores = softmax_backward(d_attention_weights, this->attention_weights);

  // 5. Invertir escalamiento y Q @ K^T.
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(this->head_dim));
  if (d_scores.isContiguous()) {
    float *d_scores_data = d_scores.getData();
#pragma omp parallel for
    for (size_t i = 0; i < d_scores.getSize(); ++i)
      d_scores_data[i] *= scale_factor;
  }

  Tensor dQ = batchMatrixMultiply(d_scores, this->k_split);
  Tensor Q_T = this->q_split.transpose(1, 2);
  Tensor dK_transposed = batchMatrixMultiply(Q_T, d_scores); // Esto es dL/dK^T
  Tensor dK = dK_transposed.transpose(1, 2);                 // Y esto es dL/dK

  // 6. Invertir division de cabezas (re-ensamblar gradientes).
  auto reassemble_grads = [&](Tensor &g) {
    g = g.reshape({B, this->num_heads, N, this->head_dim});
    g = g.transpose(1, 2).contiguous();
    return g.reshape({B, N, this->embedding_dim});
  };
  dQ = reassemble_grads(dQ);
  dK = reassemble_grads(dK);
  dV = reassemble_grads(dV);

  // 7. Invertir proyecciones de entrada (Q, K, V).
  Tensor dInput_q = this->q_proj->backward(dQ);
  Tensor dInput_k = this->k_proj->backward(dK);
  Tensor dInput_v = this->v_proj->backward(dV);

  // 8. Sumar gradientes de las 3 ramas.
  return dInput_q + dInput_k + dInput_v;
}

std::vector<Tensor *> MultiHeadAttention::getParameters() {
  auto q_params = q_proj->getParameters();
  auto k_params = k_proj->getParameters();
  auto v_params = v_proj->getParameters();
  auto out_params = out_proj->getParameters();

  std::vector<Tensor *> all_params;
  all_params.insert(all_params.end(), q_params.begin(), q_params.end());
  all_params.insert(all_params.end(), k_params.begin(), k_params.end());
  all_params.insert(all_params.end(), v_params.begin(), v_params.end());
  all_params.insert(all_params.end(), out_params.begin(), out_params.end());
  return all_params;
}

std::vector<Tensor *> MultiHeadAttention::getGradients() {
  auto q_grads = q_proj->getGradients();
  auto k_grads = k_proj->getGradients();
  auto v_grads = v_proj->getGradients();
  auto out_grads = out_proj->getGradients();

  std::vector<Tensor *> all_grads;
  all_grads.insert(all_grads.end(), q_grads.begin(), q_grads.end());
  all_grads.insert(all_grads.end(), k_grads.begin(), k_grads.end());
  all_grads.insert(all_grads.end(), v_grads.begin(), v_grads.end());
  all_grads.insert(all_grads.end(), out_grads.begin(), out_grads.end());
  return all_grads;
}

// --- Implementacion de Softmax (deberia ir a un archivo de utilidades/activaciones) ---
Tensor softmax(const Tensor &logits, int axis) {
  const auto &shape = logits.getShape();
  if (axis < 0)
    axis = shape.size() + axis;

  Tensor probabilities(shape);

  if (axis == 2 && shape.size() == 3) { // Nuestro caso de uso
#pragma omp parallel for collapse(2)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t n = 0; n < shape[1]; ++n) {
        // Truco de estabilidad: restar el maximo.
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t d = 0; d < shape[2]; ++d) {
          if (logits(b, n, d) > max_logit)
            max_logit = logits(b, n, d);
        }

        float sum_exp = 0.0f;
        for (size_t d = 0; d < shape[2]; ++d) {
          float exp_val = std::exp(logits(b, n, d) - max_logit);
          probabilities(b, n, d) = exp_val;
          sum_exp += exp_val;
        }

        for (size_t d = 0; d < shape[2]; ++d) {
          probabilities(b, n, d) /= sum_exp;
        }
      }
    }
  } else {
    throw std::runtime_error("Softmax en este eje no esta implementado.");
  }
  return probabilities;
}

Tensor softmax_backward(const Tensor &grad_output, const Tensor &softmax_output) {
  const auto &shape = grad_output.getShape();
  Tensor grad_input(shape); // dL/dZ (gradiente de los logits)

  if (shape.size() == 3) {
#pragma omp parallel for collapse(2)
    for (size_t b = 0; b < shape[0]; ++b) {
      for (size_t n = 0; n < shape[1]; ++n) {
        // Para cada fila de la matriz de atencion:
        // dL/dZ_i = S_i * (dL/dS_i - sum(dL/dS_j * S_j))
        float dot_product = 0.0f;
        for (size_t k = 0; k < shape[2]; ++k) {
          dot_product += grad_output(b, n, k) * softmax_output(b, n, k);
        }

        for (size_t i = 0; i < shape[2]; ++i) {
          float s_i = softmax_output(b, n, i);
          grad_input(b, n, i) = s_i * (grad_output(b, n, i) - dot_product);
        }
      }
    }
  } else {
    throw std::runtime_error("softmax_backward no implementado para este rank.");
  }
  return grad_input;
}

#ifndef PATCHEMBEDDING_HPP
#define PATCHEMBEDDING_HPP

#include "layers/Dense.hpp"
#include "layers/Layer.hpp"
#include <memory>

// Convierte un lote de imagenes en una secuencia de embeddings de parches.
// Realiza dos pasos:
// 1. Divide las imagenes en parches fijos.
// 2. Aplana cada parche y lo proyecta a la dimension del embedding
//    a traves de una capa lineal (Dense).
class PatchEmbedding : public Layer {
public:
  // Constructor.
  PatchEmbedding(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim);

  // Realiza el paso de parcheo y proyeccion.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Realiza el paso hacia atras a traves de la proyeccion y el "des-parcheo".
  Tensor backward(const Tensor &outputGradient) override;

  // Devuelve los parametros de la capa de proyeccion interna.
  std::vector<Tensor *> getParameters() override;

  // Devuelve los gradientes de la capa de proyeccion interna.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "PatchEmbedding"; }

  // Devuelve el numero de parches generados.
  size_t getNumPatches() const { return num_patches; }

private:
  size_t image_height, image_width, patch_size, in_channels, embedding_dim;
  size_t patch_dim;   // Dimension del parche aplanado (patch_size * patch_size * channels).
  size_t num_patches; // Numero total de parches por imagen.

  // Capa de proyeccion lineal.
  std::unique_ptr<Dense> projectionLayer;

  // Tensor con los parches aplanados, guardado para el backward pass.
  Tensor flattenedPatches;
};

#endif // PATCHEMBEDDING_HPP

#include "layers/PatchEmbedding.hpp"
#include <stdexcept>

PatchEmbedding::PatchEmbedding(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels,
                               size_t embedding_dim)
    : image_height(image_height), image_width(image_width), patch_size(patch_size), in_channels(in_channels),
      embedding_dim(embedding_dim) {

  if (image_height % patch_size != 0 || image_width % patch_size != 0) {
    throw std::invalid_argument("Las dimensiones de la imagen deben ser divisibles por el tamaño del parche.");
  }

  size_t num_patches_h = image_height / patch_size;
  size_t num_patches_w = image_width / patch_size;
  this->num_patches = num_patches_h * num_patches_w;
  this->patch_dim = patch_size * patch_size * in_channels;

  // Inicializa la capa de proyeccion lineal.
  this->projectionLayer = std::make_unique<Dense>(this->patch_dim, this->embedding_dim);
}

Tensor PatchEmbedding::forward(const Tensor &input, bool isTraining) {
  const auto &inputShape = input.getShape();
  size_t batchSize = inputShape[0];

  // Tensor para almacenar los parches aplanados, listo para la capa Densa.
  Tensor patches_flat({batchSize * this->num_patches, this->patch_dim});

  size_t patch_index_global = 0;
  // Itera sobre cada imagen en el batch.
  for (size_t b = 0; b < batchSize; ++b) {
    // Itera para extraer cada parche.
    for (size_t ph = 0; ph < image_height / patch_size; ++ph) {
      for (size_t pw = 0; pw < image_width / patch_size; ++pw) {
        // Extrae el parche como una vista, sin copiar datos.
        size_t h_start = ph * patch_size;
        size_t w_start = pw * patch_size;
        Tensor patch_view = input
                                .slice(0, b, 1)                 // Vista de la imagen b
                                .slice(2, h_start, patch_size)  // Vista de la fila de parches
                                .slice(3, w_start, patch_size); // Vista del parche exacto

        // Copia la vista del parche aplanado a nuestro tensor de parches.
        // Asumimos que la vista del parche es contigua.
        const float *patch_data = patch_view.getData() + patch_view.getDataOffset();
        float *dest_data = patches_flat.getData() + (patch_index_global * this->patch_dim);
        std::copy(patch_data, patch_data + this->patch_dim, dest_data);

        patch_index_global++;
      }
    }
  }

  if (isTraining) {
    this->flattenedPatches = patches_flat;
  }

  // Proyecta los parches aplanados al espacio de embedding.
  Tensor projected_patches = this->projectionLayer->forward(patches_flat, isTraining);
  return projected_patches.reshape({batchSize, this->num_patches, this->embedding_dim});
}

Tensor PatchEmbedding::backward(const Tensor &outputGradient) {
  const auto &gradShape = outputGradient.getShape();
  size_t batchSize = gradShape[0];

  // 1. Propagar el gradiente hacia atras a traves de la capa de proyeccion.
  Tensor grad2D = outputGradient.reshape({batchSize * this->num_patches, this->embedding_dim});
  Tensor patch_gradient = this->projectionLayer->backward(grad2D); // -> {B*num_patches, patch_dim}

  // 2. "Des-parchear" el gradiente, escribiendolo de vuelta en la forma de la imagen.
  Tensor input_gradient({batchSize, this->in_channels, this->image_height, this->image_width});
  input_gradient.fill(0.0f);

  size_t patch_index_global = 0;
  for (size_t b = 0; b < batchSize; ++b) {
    for (size_t ph = 0; ph < image_height / patch_size; ++ph) {
      for (size_t pw = 0; pw < image_width / patch_size; ++pw) {
        // Vista del gradiente del parche actual.
        Tensor current_patch_grad = patch_gradient.slice(0, patch_index_global, 1);
        const float *grad_data = current_patch_grad.getData();

        // Escribe el gradiente del parche de vuelta en su posicion original.
        for (size_t c = 0; c < this->in_channels; ++c) {
          for (size_t h = 0; h < this->patch_size; ++h) {
            for (size_t w = 0; w < this->patch_size; ++w) {
              size_t grad_idx = c * (this->patch_size * this->patch_size) + h * this->patch_size + w;
              input_gradient(b, c, ph * this->patch_size + h, pw * this->patch_size + w) = grad_data[grad_idx];
            }
          }
        }
        patch_index_global++;
      }
    }
  }
  return input_gradient;
}

std::vector<Tensor *> PatchEmbedding::getParameters() { return this->projectionLayer->getParameters(); }

std::vector<Tensor *> PatchEmbedding::getGradients() { return this->projectionLayer->getGradients(); }

#ifndef EMBEDDINGS_HPP
#define EMBEDDINGS_HPP

#include "layers/Layer.hpp"
#include "layers/PatchEmbedding.hpp"
#include <memory>

// Encapsula toda la logica de preparacion de la entrada para un ViT.
// Realiza tres operaciones clave:
// 1. Usa PatchEmbedding para convertir imagenes en embeddings de parches.
// 2. Pre-añade un token de clasificacion [CLS] entrenable a la secuencia.
// 3. Suma una codificacion posicional entrenable a la secuencia combinada.
class Embeddings : public Layer {
public:
  // Constructor.
  Embeddings(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim);

  // Realiza el paso hacia adelante.
  Tensor forward(const Tensor &input, bool isTraining) override;

  // Realiza el paso hacia atras.
  Tensor backward(const Tensor &outputGradient) override;

  // Recolecta los parametros de PatchEmbedding y los propios (CLS, Positional).
  std::vector<Tensor *> getParameters() override;

  // Recolecta los gradientes de PatchEmbedding y los propios.
  std::vector<Tensor *> getGradients() override;

  // Devuelve el nombre de la capa.
  std::string getName() const override { return "Embeddings"; }

private:
  // Capa de parcheo contenida.
  std::unique_ptr<PatchEmbedding> patcher;

  // Parametros entrenables propios de esta capa
  Tensor clsToken;           // Forma {1, 1, embedding_dim}.
  Tensor positionalEncoding; // Forma {1, num_patches + 1, embedding_dim}.

  // Gradientes correspondientes
  Tensor clsTokenGradient;
  Tensor positionalEncodingGradient;

  // Dimensiones guardadas
  size_t num_patches;
  size_t embedding_dim;
};

#endif // EMBEDDINGS_HPP

#include "core/Tensor.hpp"
#include "layers/Embeddings.hpp"

Embeddings::Embeddings(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim)
    : embedding_dim(embedding_dim) {

  // Inicializa la capa de parcheo interna.
  patcher = std::make_unique<PatchEmbedding>(image_height, image_width, patch_size, in_channels, embedding_dim);
  this->num_patches = patcher->getNumPatches();

  // Inicializa los parametros entrenables con valores pequeños aleatorios.
  float stddev = 0.02f;
  clsToken = Tensor({1, 1, embedding_dim});
  clsToken.randomizeNormal(0.0f, stddev);

  positionalEncoding = Tensor({1, 1 + this->num_patches, embedding_dim});
  positionalEncoding.randomizeNormal(0.0f, stddev);

  // Inicializa los gradientes con la misma forma, a cero.
  clsTokenGradient = Tensor(clsToken.getShape());
  positionalEncodingGradient = Tensor(positionalEncoding.getShape());
}

Tensor Embeddings::forward(const Tensor &input, bool isTraining) {
  size_t batchSize = input.getShape()[0];

  // 1. Obtener los embeddings de los parches.
  Tensor patch_embeddings = this->patcher->forward(input, isTraining); // -> {B, N, D}

  // 2. Expandir el token CLS para que coincida con el tamaño del batch.
  // expand() crea una vista {B, 1, D} sin copiar memoria.
  Tensor cls_token_expanded = expand(this->clsToken, 0, batchSize);

  // 3. Concatenar el CLS token y los parches a lo largo del eje de la secuencia.
  Tensor embeddings_with_cls = concatenate({cls_token_expanded, patch_embeddings}, 1); // -> {B, N+1, D}

  // 4. Añadir la codificacion posicional por broadcasting.
  embeddings_with_cls.addBroadcast(this->positionalEncoding);

  return embeddings_with_cls;
}

Tensor Embeddings::backward(const Tensor &outputGradient) {
  // El gradiente de una suma es el mismo para ambas ramas.
  // Por tanto, el gradiente de la codificacion posicional es la suma a traves del batch.
  this->positionalEncodingGradient = outputGradient.sum(0); // -> {1, N+1, D}

  // El gradiente que fluye hacia la concatenacion es el mismo outputGradient.
  Tensor grad_before_pos = outputGradient;

  // "Des-concatenar" el gradiente obteniendo vistas (slices).
  Tensor grad_cls = grad_before_pos.slice(1, 0, 1);               // -> {B, 1, D}
  Tensor grad_patches = grad_before_pos.slice(1, 1, num_patches); // -> {B, N, D}

  // El gradiente del token CLS es la suma a traves del batch de su gradiente.
  this->clsTokenGradient = grad_cls.sum(0); // -> {1, 1, D}

  // El gradiente de los parches debe ser contiguo para pasarlo a la siguiente capa.
  Tensor grad_patches_contiguous = grad_patches.contiguous();

  // Propagar el gradiente de los parches a la capa de parcheo.
  Tensor input_image_gradient = this->patcher->backward(grad_patches_contiguous);

  return input_image_gradient;
}

std::vector<Tensor *> Embeddings::getParameters() {
  // Obtiene los parametros de la capa interna.
  auto params = this->patcher->getParameters();
  // Añade los parametros propios de esta capa.
  params.push_back(&this->clsToken);
  params.push_back(&this->positionalEncoding);
  return params;
}

std::vector<Tensor *> Embeddings::getGradients() {
  auto grads = this->patcher->getGradients();
  grads.push_back(&this->clsTokenGradient);
  grads.push_back(&this->positionalEncodingGradient);
  return grads;
}

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "core/Tensor.hpp"
#include <vector>

// Clase base abstracta para todos los algoritmos de optimizacion.
// Define la interfaz para optimizadores como SGD o Adam, cuya tarea es
// actualizar los parametros de la red usando los gradientes calculados.
class Optimizer {
public:
  // Constructor que define la tasa de aprendizaje.
  explicit Optimizer(float learningRate) : learningRate(learningRate) {}

  // Destructor virtual para herencia polimorfica.
  virtual ~Optimizer() = default;

  // Realiza un unico paso de optimizacion para actualizar los parametros.
  // - parameters: Punteros a los parametros entrenables del modelo.
  // - gradients: Punteros a los gradientes correspondientes a cada parametro.
  virtual void update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) = 0;

protected:
  // Tasa de aprendizaje (learning rate) del algoritmo.
  float learningRate;
};

#endif // OPTIMIZER_HPP

#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizers/Optimizer.hpp"
#include <vector>

// Implementa el optimizador Adam (Adaptive Moment Estimation).
// Adam combina las ideas de Momentum (primer momento) y RMSprop (segundo momento)
// para adaptar la tasa de aprendizaje para cada parametro individual.
class Adam : public Optimizer {
public:
  // Constructor para el optimizador Adam.
  Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8, float weight_decay = 0.0f);

  // Realiza un unico paso de actualizacion de Adam.
  void update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) override;

private:
  // Hiperparametros de Adam.
  float beta1;
  float beta2;
  float epsilon;
  float weight_decay; // Termino de regularizacion L2 (Weight Decay).
  long long t;        // Contador de pasos de tiempo para la correccion de sesgo.

  // Estado del optimizador (se inicializan en la primera llamada a update).
  std::vector<Tensor> m; // Estimacion del primer momento (media de gradientes).
  std::vector<Tensor> v; // Estimacion del segundo momento (media de gradientes^2).

  // Flag para la inicializacion diferida de los tensores de momento.
  bool initialized;
};

#endif // ADAM_HPP
//

#include "optimizers/Adam.hpp"
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

Adam::Adam(float learningRate, float beta1, float beta2, float epsilon, float weight_decay)
    : Optimizer(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), weight_decay(weight_decay), t(0),
      initialized(false) {}

void Adam::update(std::vector<Tensor *> &parameters, const std::vector<Tensor *> &gradients) {
  if (parameters.size() != gradients.size()) {
    throw std::runtime_error("El numero de parametros y gradientes no coincide en Adam::update.");
  }

  // Inicializacion diferida: crea los tensores de momento m y v en la primera llamada.
  if (!initialized) {
    m.reserve(parameters.size());
    v.reserve(parameters.size());
    for (const auto &param : parameters) {
      m.emplace_back(param->getShape());
      v.emplace_back(param->getShape());
    }
    initialized = true;
  }

  t++; // Incrementa el contador de pasos.

  // Correccion de sesgo (bias correction) pre-calculada.
  const float beta1_t = std::pow(beta1, t);
  const float beta2_t = std::pow(beta2, t);

  // Itera sobre cada par de parametro/gradiente.
  for (size_t i = 0; i < parameters.size(); ++i) {
    Tensor *param = parameters[i];
    const Tensor *grad_tensor = gradients[i];
    Tensor &m_i = m[i];
    Tensor &v_i = v[i];

    const auto &shape = param->getShape();

    // Actualizacion de parametros, con bucles especializados por dimensionalidad.
    if (shape.size() == 1 || shape.size() == 2 || shape.size() == 3) {
      if (shape.size() == 1) { // Para Bias, LayerNorm
#pragma omp parallel for
        for (size_t j = 0; j < shape[0]; ++j) {
          float g = (*grad_tensor)(j);
          // Añade el decaimiento de peso (L2 regularization) al gradiente.
          // No se suele aplicar a los biases ni a los parametros de LayerNorm.
          // (Aqui lo aplicamos a todos por simplicidad, se podria refinar).
          if (weight_decay > 0.0f) {
            g += weight_decay * (*param)(j);
          }
          m_i(j) = beta1 * m_i(j) + (1.0f - beta1) * g;
          v_i(j) = beta2 * v_i(j) + (1.0f - beta2) * (g * g);
          float m_hat = m_i(j) / (1.0f - beta1_t);
          float v_hat = v_i(j) / (1.0f - beta2_t);
          (*param)(j) -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
      } else if (shape.size() == 2) { // Para pesos de Dense
#pragma omp parallel for collapse(2)
        for (size_t r = 0; r < shape[0]; ++r) {
          for (size_t c = 0; c < shape[1]; ++c) {
            float g = (*grad_tensor)(r, c);
            if (weight_decay > 0.0f) {
              g += weight_decay * (*param)(r, c);
            }
            m_i(r, c) = beta1 * m_i(r, c) + (1.0f - beta1) * g;
            v_i(r, c) = beta2 * v_i(r, c) + (1.0f - beta2) * (g * g);
            float m_hat = m_i(r, c) / (1.0f - beta1_t);
            float v_hat = v_i(r, c) / (1.0f - beta2_t);
            (*param)(r, c) -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
          }
        }
      } else { // Para embeddings posicionales, etc. (3D)
#pragma omp parallel for collapse(3)
        for (size_t d0 = 0; d0 < shape[0]; ++d0) {
          for (size_t d1 = 0; d1 < shape[1]; ++d1) {
            for (size_t d2 = 0; d2 < shape[2]; ++d2) {
              float g = (*grad_tensor)(d0, d1, d2);
              if (weight_decay > 0.0f) {
                g += weight_decay * (*param)(d0, d1, d2);
              }
              m_i(d0, d1, d2) = beta1 * m_i(d0, d1, d2) + (1.0f - beta1) * g;
              v_i(d0, d1, d2) = beta2 * v_i(d0, d1, d2) + (1.0f - beta2) * (g * g);
              float m_hat = m_i(d0, d1, d2) / (1.0f - beta1_t);
              float v_hat = v_i(d0, d1, d2) / (1.0f - beta2_t);
              (*param)(d0, d1, d2) -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
          }
        }
      }
    }
  }
}

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
