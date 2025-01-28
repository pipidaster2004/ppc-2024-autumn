// Copyright 2024 Nesterov Alexander
#include "seq/sorokin_a_check_lexicographic_order_of_strings/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool sorokin_a_check_lexicographic_order_of_strings_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<std::vector<char>>(taskData->inputs_count[0], std::vector<char>(taskData->inputs_count[1]));

  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  res_ = 0;
  return true;
}

bool sorokin_a_check_lexicographic_order_of_strings_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == 1;
}

bool sorokin_a_check_lexicographic_order_of_strings_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < std::min(input_[0].size(), input_[1].size()); ++i) {
    if (static_cast<int>(input_[0][i]) > static_cast<int>(input_[1][i])) {
      res_ = 1;
      break;
    }
    if (static_cast<int>(input_[0][i]) < static_cast<int>(input_[1][i])) {
      break;
    }
  }
  return true;
}

bool sorokin_a_check_lexicographic_order_of_strings_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}
