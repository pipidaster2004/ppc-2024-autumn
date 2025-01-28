#include "seq/khokhlov_a_sum_values_by_rows/include/ops_sec.hpp"

using namespace std::chrono_literals;

bool khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto tmp = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0], input_.begin());
  row = taskData->inputs_count[1];
  col = taskData->inputs_count[2];
  // Init value for output
  sum = std::vector<int>(row, 0);
  return true;
}

bool khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 && taskData->inputs_count[2] >= 0 &&
          taskData->inputs_count[1] == taskData->outputs_count[0]);
}

bool khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows::run() {
  internal_order_test();
  for (int i = 0; i < row; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < col; j++) tmp_sum += input_[i * col + j];
    sum[i] += tmp_sum;
  }
  return true;
}

bool khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows::post_processing() {
  internal_order_test();
  for (int i = 0; i < row; i++) reinterpret_cast<int*>(taskData->outputs[0])[i] = sum[i];
  return true;
}

std::vector<int> khokhlov_a_sum_values_by_rows_seq::getRandomMatrix(int rows, int cols) {
  int sz = rows * cols;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}