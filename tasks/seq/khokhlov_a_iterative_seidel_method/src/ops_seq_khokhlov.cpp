#include "seq/khokhlov_a_iterative_seidel_method/include/ops_seq_khokhlov.hpp"

bool khokhlov_a_iterative_seidel_method_seq::seidel_method_seq::pre_processing() {
  internal_order_test();
  // init matrix
  A = std::vector<double>(taskData->inputs_count[0] * taskData->inputs_count[0]);
  auto tmp = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0] * taskData->inputs_count[0], A.begin());

  // init vector
  b = std::vector<double>(taskData->inputs_count[0]);
  auto tmp1 = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(tmp1, tmp1 + taskData->inputs_count[0], b.begin());

  // init sizes
  n = taskData->inputs_count[0];

  // init maxIterations
  maxIterations = taskData->inputs_count[1];

  // Init value for output
  result = std::vector<double>(taskData->inputs_count[0], 0);
  return true;
}
bool khokhlov_a_iterative_seidel_method_seq::seidel_method_seq::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0);
}

bool khokhlov_a_iterative_seidel_method_seq::seidel_method_seq::run() {
  internal_order_test();
  std::vector<double> x(n, 1.0);
  std::vector<double> prevX(n, 1.0);

  for (int iter = 0; iter < maxIterations; ++iter) {
    for (int i = 0; i < n; ++i) {
      double sum = 0;
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          sum += (A[i * n + j] * x[j]);
        }
      }
      if (A[i * n + i] != 0) {
        x[i] = (b[i] - sum) / A[i * n + i];
      } else {
        x[i] = 0.0;
      }
    }
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
      norm += std::pow(x[i] - prevX[i], 2);
    }
    norm = std::sqrt(norm);

    if (norm < EPSILON) {
      break;
    }
    prevX = x;
  }
  result = x;
  return true;
}

bool khokhlov_a_iterative_seidel_method_seq::seidel_method_seq::post_processing() {
  internal_order_test();
  for (int i = 0; i < n; i++) reinterpret_cast<double*>(taskData->outputs[0])[i] = result[i];
  return true;
}

void khokhlov_a_iterative_seidel_method_seq::getRandomSLAU(std::vector<double>& A, std::vector<double>& b, int N) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < N; ++i) {
    double rowSum = 0.0;
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        A[i * N + j] = rand() % 10 - 5;
        rowSum += std::abs(A[i * N + j]);
      }
    }
    A[i * N + i] = rowSum + (rand() % 5 + 1);
    b[i] = rand() % 20 - 10;
  }
}