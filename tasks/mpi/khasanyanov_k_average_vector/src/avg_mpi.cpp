#include "mpi/khasanyanov_k_average_vector/include/avg_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

std::vector<int> khasanyanov_k_average_vector_mpi::getRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = gen() % 1000;
  }
  return vec;
}
