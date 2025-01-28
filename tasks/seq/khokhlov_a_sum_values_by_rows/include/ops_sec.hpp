// Copyright 2023 Nesterov Alexander
#pragma once

#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_sum_values_by_rows_seq {

std::vector<int> getRandomMatrix(int rows, int cols);

class Sum_val_by_rows : public ppc::core::Task {
 public:
  explicit Sum_val_by_rows(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int row, col;
  std::vector<int> sum;
};

}  // namespace khokhlov_a_sum_values_by_rows_seq