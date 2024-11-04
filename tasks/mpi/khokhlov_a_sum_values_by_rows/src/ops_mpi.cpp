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
  int delta = 0;
  if (world.rank() == 0) {
    delta = (taskData->inputs_count[1] / world.size());
    delta += (taskData->inputs_count[1] % world.size() == 0) ? 0 : 1;
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta * col, col*delta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta * col);
  } else {
    world.recv(0, 0, local_input_.data(), col*delta);
  }
  std::vector<int> local_sum;
  for (int i = 0; i < delta; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < col; j++) {
      tmp_sum += local_input_[i * col + j];
    }
    local_sum.push_back(tmp_sum);
  }
  if (world.rank() == 0) {
    std::vector<int> local_res(row + delta * world.size());
    std::vector<int> sizes(world.size(), col*delta);
    boost::mpi::gatherv(world, local_sum.data(), local_sum.size(), local_res.data(), sizes, 0);
    local_res.resize(row);
    sum = local_res;
  } else {
    boost::mpi::gatherv(world, local_sum.data(), local_sum.size(), 0);
  }
  return true;
} 

bool khokhlov_a_sum_values_by_rows_mpi::Sum_val_by_rows_mpi::post_processing() {
  internal_order_test();
  if (world.rank() == 0)
    for (int i = 0; i < row; i++) reinterpret_cast<int*>(taskData->outputs[0])[i] = sum[i];
  return true;
}

std::vector<int> khokhlov_a_sum_values_by_rows_mpi::getRandomMatrix(int rows, int cols) {
  int sz = rows * cols;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}