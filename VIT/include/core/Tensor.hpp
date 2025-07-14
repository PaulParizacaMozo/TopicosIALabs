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
