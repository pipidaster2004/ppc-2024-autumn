#include "mpi/khokhlov_a_sum_values_by_rows/include/ops_mpi.hpp"

using namespace std::chrono_literals;

bool khokhlov_a_sum_values_by_rows_mpi::Sum_val_by_rows_seq::pre_processing() {
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

bool khokhlov_a_sum_values_by_rows_mpi::Sum_val_by_rows_seq::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 && taskData->inputs_count[2] >= 0 &&
          taskData->inputs_count[1] == taskData->outputs_count[0]);
}

bool khokhlov_a_sum_values_by_rows_mpi::Sum_val_by_rows_seq::run() {
  internal_order_test();
  for (int i = 0; i < row; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < col; j++) {
      tmp_sum += input_[i * col + j];
    }
    sum[i] += tmp_sum;
  }
  return true;
}

bool khokhlov_a_sum_values_by_rows_mpi::Sum_val_by_rows_seq::post_processing() {
  internal_order_test();
  for (int i = 0; i < row; i++) reinterpret_cast<int*>(taskData->outputs[0])[i] = sum[i];
  return true;
}

bool khokhlov_a_sum_values_by_rows_mpi::Sum_val_by_rows_mpi::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto tmp = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp, tmp + taskData->inputs_count[0], input_.begin());
    row = taskData->inputs_count[1];
    col = taskData->inputs_count[2];
    // Init value for output
    sum = std::vector<int>(row, 0);
  }
  return true;
}

bool khokhlov_a_sum_values_by_rows_mpi::Sum_val_by_rows_mpi::validation() {
  internal_order_test();
  if (world.rank() == 0)
    return (taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 && taskData->inputs_count[2] >= 0 &&
            taskData->inputs_count[1] == taskData->outputs_count[0]);
  return true;
}

bool khokhlov_a_sum_values_by_rows_mpi::Sum_val_by_rows_mpi::run() {
  internal_order_test();
  broadcast(world, row, 0);
  broadcast(world, col, 0);

  int delta = row / world.size();
  int last_row = row % world.size();
  int local_n = (world.rank() == world.size() - 1) ? delta + last_row : delta;

  local_input_ = std::vector<int>(local_n * col);
  std::vector<int> send_counts_(world.size());
  std::vector<int> recv_counts_(world.size());
  for (int i = 0; i < world.size(); ++i) {
    send_counts_[i] = (i == world.size() - 1) ? delta + last_row : delta;
    send_counts_[i] *= col;
    recv_counts_[i] = (i == world.size() - 1) ? delta + last_row : delta;
  }
  boost::mpi::scatterv(world, input_.data(), send_counts_, local_input_.data(), 0);

  std::vector<int> local_sum(local_n, 0);
  for (int i = 0; i < local_n; ++i) {
    for (int j = 0; j < col; ++j) {
      local_sum[i] += local_input_[i * col + j];
    }
  }

  boost::mpi::gatherv(world, local_sum.data(), local_sum.size(), sum.data(), recv_counts_, 0);

  return true;
}

bool khokhlov_a_sum_values_by_rows_mpi::Sum_val_by_rows_mpi::post_processing() {
  internal_order_test();
  if (world.rank() == 0)
    for (int i = 0; i < row; i++) reinterpret_cast<int*>(taskData->outputs[0])[i] = sum[i];
  return true;
}

std::vector<int> khokhlov_a_sum_values_by_rows_mpi::getRandomMatrix(int size) {
  int sz = size;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}