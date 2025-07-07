#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @class Tensor
 * @brief Una implementación de un tensor N-dimensional para redes neuronales.
 *
 * Esta clase gestiona un bloque de datos multidimensional. Su diseño se centra en la eficiencia,
 * utilizando un puntero compartido (std::shared_ptr) para permitir "vistas" o "slices"
 * de bajo coste (sin copia de datos). La navegación por los datos se realiza mediante un
 * sistema de "strides", que mapea los índices multidimensionales a una posición en el
 * bloque de memoria 1D subyacente.
 *
 * Hay dos tipos de Tensores:
 *  1. Owning Tensor: Gestiona y es propietario de la memoria de datos.
 *  2. View Tensor: Es una vista sobre la memoria de un Owning Tensor (o de otra vista).
 *     No posee los datos, solo los referencia.
 */
class Tensor {
public:
  // --- Constructores ---

  /** @brief Constructor por defecto. Crea un tensor vacío. */
  Tensor();

  /**
   * @brief Constructor de un "Owning Tensor" (propietario de la memoria).
   * @param shape Las dimensiones del tensor (ej: {batch, channels, height, width}).
   */
  explicit Tensor(const std::vector<size_t> &shape);

  /**
   * @brief Constructor de un "Owning Tensor" con datos iniciales.
   * @param shape Las dimensiones del tensor.
   * @param data Vector 1D con los datos iniciales en formato row-major.
   */
  Tensor(const std::vector<size_t> &shape, const std::vector<float> &data);

  // --- Constructores y asignaciones de copia/movimiento (bajo coste) ---
  Tensor(const Tensor &other) = default;
  Tensor(Tensor &&other) noexcept = default;
  Tensor &operator=(const Tensor &other) = default;
  Tensor &operator=(Tensor &&other) noexcept = default;
  ~Tensor() = default;

  // --- Operadores de Acceso a Elementos ---

  /** @brief Acceso genérico a elementos para N dimensiones. */
  template <typename... Args> float &operator()(Args... args);
  template <typename... Args> const float &operator()(Args... args) const;

  /** @brief Acceso optimizado para tensores 1D. */
  float &operator()(size_t i);
  const float &operator()(size_t i) const;

  /** @brief Acceso optimizado para tensores 2D (matrices). */
  float &operator()(size_t i, size_t j);
  const float &operator()(size_t i, size_t j) const;

  /** @brief Acceso optimizado para tensores 4D (ej: imágenes de un batch). */
  float &operator()(size_t b, size_t c, size_t h, size_t w);
  const float &operator()(size_t b, size_t c, size_t h, size_t w) const;

  // --- Operaciones y Vistas ---

  /**
   * @brief Crea una vista (slice) del tensor a lo largo de la primera dimensión.
   * @details Operación de coste muy bajo, no se copian los datos. Ideal para crear mini-batches.
   * @param start Índice inicial del slice (inclusive).
   * @param count Número de elementos a incluir en el slice.
   * @return Un nuevo Tensor que es una vista de los datos originales.
   */
  Tensor slice(size_t start, size_t count) const;

  /** @brief Devuelve un nuevo tensor que es la transpuesta del original (para 2D). */
  Tensor transpose() const;

  /** @brief Devuelve un nuevo tensor con el cuadrado de cada elemento. */
  Tensor square() const;

  /**
   * @brief Suma los elementos de un tensor a lo largo de un eje específico.
   * @param axis El eje sobre el cual se realizará la suma.
   * @return Un nuevo tensor con una dimensión menos, conteniendo el resultado de la suma.
   */
  Tensor sum(size_t axis) const;

  /**
   * @brief Realiza una suma con "broadcasting" del `other` tensor.
   * @details `other` debe tener dimensiones compatibles para broadcasting (ej: un vector de bias).
   * @param other El tensor que se sumará.
   */
  void addBroadcast(const Tensor &other);

  // --- Métodos de Inicialización (solo para "Owning Tensors") ---

  /** @brief Rellena el tensor con un valor escalar. */
  void fill(float value);

  /** @brief Inicializa el tensor con valores aleatorios en un rango. */
  void randomize(float min = -1.0f, float max = 1.0f);

  // --- Getters y Utilidades ---

  /** @brief Devuelve la forma (dimensiones) del tensor. */
  const std::vector<size_t> &getShape() const { return shape; }

  /** @brief Devuelve el número total de elementos en el tensor. */
  size_t getSize() const { return totalSize; }

  /** @brief Devuelve los strides del tensor. */
  const std::vector<size_t> &getStrides() const { return strides; }

  /** @brief Devuelve un puntero de solo lectura al inicio del bloque de datos subyacente. */
  const float *getData() const;

  /** @brief Devuelve un puntero de escritura al inicio del bloque de datos subyacente. */
  float *getData();

  /** @brief Devuelve una representación en string de la forma del tensor. */
  std::string shapeToString() const;

private:
  /**
   * @brief Constructor privado para crear vistas (slices).
   * @param dataPtr Puntero compartido al bloque de datos original.
   * @param shape La nueva forma de la vista.
   * @param strides Los strides del tensor original (se reutilizan).
   * @param offset El desplazamiento en elementos desde el inicio del `dataPtr`.
   */
  Tensor(std::shared_ptr<std::vector<float>> dataPtr, const std::vector<size_t> &shape, const std::vector<size_t> &strides,
         size_t offset);

  /** @brief Calcula los strides basándose en la forma del tensor. */
  void computeStrides();

  /** @brief Calcula el índice plano en el vector 1D a partir de índices multidimensionales. */
  template <typename... Args> size_t getFlatIndex(Args... args) const;

  // --- Miembros ---
  std::shared_ptr<std::vector<float>> dataPtr; ///< Puntero compartido a los datos. Permite vistas eficientes.
  std::vector<size_t> shape;                   ///< Dimensiones de este tensor/vista (ej: {N, C, H, W}).
  std::vector<size_t> strides;                 ///< Pasos en memoria para cada dimensión. Clave para el acceso.
  size_t dataOffset;                           ///< Desplazamiento desde el inicio de `dataPtr` para esta vista.
  size_t totalSize;                            ///< Número total de elementos en esta vista/tensor.
};

// --- Funciones Libres ---

/** @brief Realiza la multiplicación de matrices entre dos tensores 2D. */
Tensor matrixMultiply(const Tensor &a, const Tensor &b);

// == IMPLEMENTACIONES INLINE (para rendimiento) ==

// --- Implementación de acceso optimizado para 1D ---
inline float &Tensor::operator()(size_t i) {
#ifndef NDEBUG // Comprobaciones solo en modo Debug
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

// --- Implementación de acceso optimizado para 2D ---
inline float &Tensor::operator()(size_t i, size_t j) {
#ifndef NDEBUG // Comprobaciones solo en modo Debug
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

// --- Implementación de acceso optimizado para 4D ---
inline float &Tensor::operator()(size_t b, size_t c, size_t h, size_t w) {
#ifndef NDEBUG
  if (shape.size() != 4 || b >= shape[0] || c >= shape[1] || h >= shape[2] || w >= shape[3])
    throw std::out_of_range("Acceso 4D fuera de rango.");
#endif
  return (*dataPtr)[dataOffset + b * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]];
}

inline const float &Tensor::operator()(size_t b, size_t c, size_t h, size_t w) const {
#ifndef NDEBUG
  if (shape.size() != 4 || b >= shape[0] || c >= shape[1] || h >= shape[2] || w >= shape[3])
    throw std::out_of_range("Acceso 4D fuera de rango.");
#endif
  return (*dataPtr)[dataOffset + b * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]];
}

// --- Implementación de templates ---

template <typename... Args> size_t Tensor::getFlatIndex(Args... args) const {
  const size_t numArgs = sizeof...(args);
  if (numArgs != shape.size()) {
    throw std::invalid_argument("Numero de indices incorrecto para la forma del tensor.");
  }

  size_t indices[] = {static_cast<size_t>(args)...};
  size_t index = dataOffset;
  for (size_t i = 0; i < numArgs; ++i) {
    if (indices[i] >= shape[i]) {
      throw std::out_of_range("Indice fuera de rango.");
    }
    index += indices[i] * strides[i];
  }
  return index;
}

template <typename... Args> float &Tensor::operator()(Args... args) { return (*dataPtr)[getFlatIndex(args...)]; }

template <typename... Args> const float &Tensor::operator()(Args... args) const { return (*dataPtr)[getFlatIndex(args...)]; }

#endif // TENSOR_HPP
