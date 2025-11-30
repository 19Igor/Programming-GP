# Задание
Выделить на GPU массив arr из 10^9 элементов типа float и инициализировать его с помощью ядра следующим образом: arr[i] = sin((i%360)*Pi/180). Скопировать массив в память центрального процессора и посчитать ошибку err = sum_i(abs(sin((i%360)*Pi/180) - arr[i]))/10^9. Провести исследование зависимости результата от использования функций: sin, sinf, __sin. Объяснить результат. Проверить результат при использовании массива типа double.
# Код
```python
%%writefile cuda_sin_test.cu

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

#define PI M_PI

const size_t N = 1000000000ULL;  

// 1.Каждое ядро обрабатывает часть массива, вычисляя значение синуса
// для каждого индекса. Используются разные версии функции sin()
// для сравнения точности и производительности.

template <typename T>
__global__ void kernel_sin(T *arr, unsigned long long N) {
    // Вычисляем глобальный индекс потока
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        // Вычисляем sin() в радианах (переводим градусы в радианы)
        arr[i] = sin((T)(i % 360) * PI / 180.0);
}

template <typename T>
__global__ void kernel_sinf(T *arr, unsigned long long N) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        // Используем sinf() — версию функции для float
        arr[i] = sinf((T)(i % 360) * (T)PI / 180.0f);
}

template <typename T>
__global__ void kernel___sinf(T *arr, unsigned long long N) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        arr[i] = __sinf((T)(i % 360) * (T)PI / 180.0f);
}

// 2. Функция вычисления средней ошибки между результатами GPU
// и эталонным результатом, рассчитанным на CPU.

template<typename T>
T compute_error(const T* arr) {
    T err = 0.0;

    for (size_t i = 0; i < N; ++i) {
        // Эталонное значение синуса, вычисленное на CPU
        T ref = sin((i % 360) * PI / 180.0);

        // Добавляем абсолютное отклонение текущего значения от эталонного
        err += fabs(ref - (T)arr[i]);
    }

    // Возвращаем среднюю ошибку по всем элементам
    return err / N;
}

// 3. Основная функция тестирования CUDA-реализаций синуса.
// Здесь происходит:
//   - Выделение памяти на GPU и CPU,
//   - Настройка сетки и блоков потоков,
//   - Запуск трёх разных ядер,
//   - Измерение времени работы,
//   - Подсчёт средней ошибки.

template<typename T>
void test() {
    T *d_arr, *h_arr;

    // Выделяем память на хосте (CPU)
    h_arr = new T[N];

    // Выделяем память на устройстве (GPU)
    cudaMalloc(&d_arr, N * sizeof(T));

    // Определяем количество потоков в блоке и количество блоков в сетке
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Тест №1 — стандартная функция sin()
    {
        std::cout << "Running sin() ..." << std::endl;

        // Засекаем время начала вычислений
        auto start = std::chrono::steady_clock::now();

        // Запуск ядра на GPU
        kernel_sin<<<grid, block>>>(d_arr, N);

        // Дожидаемся завершения всех потоков на устройстве
        cudaDeviceSynchronize();

        // Засекаем время окончания вычислений
        auto stop = std::chrono::steady_clock::now();

        // Копируем результаты из памяти GPU в память CPU
        cudaMemcpy(h_arr, d_arr, N * sizeof(T), cudaMemcpyDeviceToHost);

        // Вычисляем и выводим время выполнения в миллисекундах
        std::cout << "Time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                  << " ms" << std::endl;

        // Вычисляем среднюю ошибку и выводим её на экран
        double err = compute_error(h_arr);
        std::cout << "Average error: " << err << std::endl << std::endl;
    }

    // Тест №2 — sinf() (float-версия функции)
    {
        std::cout << "Running sinf() ..." << std::endl;

        auto start = std::chrono::steady_clock::now();
        kernel_sinf<<<grid, block>>>(d_arr, N);
        cudaDeviceSynchronize();
        auto stop = std::chrono::steady_clock::now();

        cudaMemcpy(h_arr, d_arr, N * sizeof(T), cudaMemcpyDeviceToHost);

        std::cout << "Time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                  << " ms" << std::endl;

        double err = compute_error(h_arr);
        std::cout << "Average error: " << err << std::endl << std::endl;
    }

    // Тест №3 — __sinf() (встроенная быстрая функция CUDA)
    {
        std::cout << "Running __sinf() ..." << std::endl;

        auto start = std::chrono::steady_clock::now();
        kernel___sinf<<<grid, block>>>(d_arr, N);
        cudaDeviceSynchronize();
        auto stop = std::chrono::steady_clock::now();

        cudaMemcpy(h_arr, d_arr, N * sizeof(T), cudaMemcpyDeviceToHost);

        std::cout << "Time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                  << " ms" << std::endl;

        double err = compute_error(h_arr);
        std::cout << "Average error: " << err << std::endl << std::endl;
    }

    cudaFree(d_arr);
    delete[] h_arr;
}

int main() {
    std::cout << "=== CUDA Sine Accuracy Test ===" << std::endl;
    std::cout << "Elements count: " << N << std::endl;

    std::cout << "\n--- FLOAT MODE ---" << std::endl;
    test<float>();

    std::cout << "\n--- DOUBLE MODE ---" << std::endl;
    test<double>();

    std::cout << "\nExperiment completed." << std::endl;
    return 0;
}


Output:
=== Тест точности функции синуса CUDA ===
Количество элементов: 1000000000

--- РЕЖИМ FLOAT ---
Выполняется sin() ...
Время выполнения: 418 мс
Средняя ошибка: 0

Выполняется sinf() ...
Время выполнения: 16 мс
Средняя ошибка: 1.6e-08

Выполняется __sinf() ...
Время выполнения: 16 мс
Средняя ошибка: 1.6e-08


--- РЕЖИМ DOUBLE ---
Выполняется sin() ...
Время выполнения: 300 мс
Средняя ошибка: 8.77963e-18

Выполняется sinf() ...
Время выполнения: 303 мс
Средняя ошибка: 1.30149e-07

Выполняется __sinf() ...
Время выполнения: 290 мс
Средняя ошибка: 1.30149e-07


Эксперимент завершён.
```
# Параметры машины
![[Pasted image 20251103152933.png|700]]
# Метрики
```python
%%writefile cuda_specs.cu

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Get the number of CUDA-capable devices

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop{}; // Initialize a cudaDeviceProp structure
        cudaGetDeviceProperties(&prop, i); // Get properties for device 'i'

        std::cout << "--- Device Number: " << i << " ---" << std::endl;
        std::cout << "  Device Name: " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory (bytes): " << prop.totalGlobalMem << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Clock Rate (kHz): " << prop.clockRate << std::endl;
        std::cout << "  Shared Memory per Block (bytes): " << prop.sharedMemPerBlock << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
    }
}


Output:
--- Device Number: 0 ---
  Device Name: Tesla T4
  Compute Capability: 7.5
  Total Global Memory (bytes): 15828320256
  Max Threads per Block: 1024
  Multiprocessor Count: 40
  Clock Rate (kHz): 1590000
  Shared Memory per Block (bytes): 49152
  Warp Size: 32
  ECC Enabled: Yes
```

