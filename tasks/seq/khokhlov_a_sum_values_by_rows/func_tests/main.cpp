#include <gtest/gtest.h>

#include "seq/khokhlov_a_sum_values_by_rows/include/ops_sec.hpp"

TEST(khokhlov_a_sum_values_by_rows_seq, validation_test) {
  const int rows = 1;
  const int cols = 1;

  // create data
  std::vector<int> in(rows * cols, 0);
  std::vector<int> expect(rows, 0);
  std::vector<int> out(rows, 0);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskdataSeq->inputs_count.emplace_back(in.size());
  taskdataSeq->inputs_count.emplace_back(rows);
  taskdataSeq->inputs_count.emplace_back(cols);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskdataSeq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows sum_val_by_rows(taskdataSeq);
  ASSERT_TRUE(sum_val_by_rows.validation());
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_empty) {
  const int rows = 0;
  const int cols = 0;

  // create data
  std::vector<int> in = {};
  std::vector<int> expect;
  std::vector<int> out = {};

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskdataSeq->inputs_count.emplace_back(in.size());
  taskdataSeq->inputs_count.emplace_back(rows);
  taskdataSeq->inputs_count.emplace_back(cols);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskdataSeq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows sum_val_by_rows(taskdataSeq);
  ASSERT_TRUE(sum_val_by_rows.validation());
  sum_val_by_rows.pre_processing();
  sum_val_by_rows.run();
  sum_val_by_rows.post_processing();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_2x2_matrix) {
  const int rows = 2;
  const int cols = 2;

  // create data
  std::vector<int> in = {1, 2, 3, 4};
  std::vector<int> expect = {3, 7};
  std::vector<int> out(rows, 0);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskdataSeq->inputs_count.emplace_back(in.size());
  taskdataSeq->inputs_count.emplace_back(rows);
  taskdataSeq->inputs_count.emplace_back(cols);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskdataSeq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows sum_val_by_rows(taskdataSeq);
  ASSERT_TRUE(sum_val_by_rows.validation());
  sum_val_by_rows.pre_processing();
  sum_val_by_rows.run();
  sum_val_by_rows.post_processing();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_2x4_matrix) {
  const int rows = 2;
  const int cols = 4;

  // create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> expect = {10, 26};
  std::vector<int> out(rows, 0);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskdataSeq->inputs_count.emplace_back(in.size());
  taskdataSeq->inputs_count.emplace_back(rows);
  taskdataSeq->inputs_count.emplace_back(cols);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskdataSeq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows sum_val_by_rows(taskdataSeq);
  ASSERT_TRUE(sum_val_by_rows.validation());
  sum_val_by_rows.pre_processing();
  sum_val_by_rows.run();
  sum_val_by_rows.post_processing();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_4x2_matrix) {
  const int rows = 4;
  const int cols = 2;

  // create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> expect = {3, 7, 11, 15};
  std::vector<int> out(rows, 0);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskdataSeq->inputs_count.emplace_back(in.size());
  taskdataSeq->inputs_count.emplace_back(rows);
  taskdataSeq->inputs_count.emplace_back(cols);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskdataSeq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows sum_val_by_rows(taskdataSeq);
  ASSERT_TRUE(sum_val_by_rows.validation());
  sum_val_by_rows.pre_processing();
  sum_val_by_rows.run();
  sum_val_by_rows.post_processing();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_4x3_matrix_with_negative_elements) {
  const int rows = 4;
  const int cols = 3;

  // create data
  std::vector<int> in = {1, 2, -3, 3, 4, -6, 5, 6, -9, 7, 8, -12};
  std::vector<int> expect = {0, 1, 2, 3};
  std::vector<int> out(rows, 0);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskdataSeq->inputs_count.emplace_back(in.size());
  taskdataSeq->inputs_count.emplace_back(rows);
  taskdataSeq->inputs_count.emplace_back(cols);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskdataSeq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows sum_val_by_rows(taskdataSeq);
  ASSERT_TRUE(sum_val_by_rows.validation());
  sum_val_by_rows.pre_processing();
  sum_val_by_rows.run();
  sum_val_by_rows.post_processing();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_100x100_matrix) {
  const int rows = 100;
  const int cols = 100;

  // create data
  std::vector<int> in(cols * rows, 2);
  std::vector<int> expect(rows, cols * 2);
  std::vector<int> out(rows, 0);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskdataSeq->inputs_count.emplace_back(in.size());
  taskdataSeq->inputs_count.emplace_back(rows);
  taskdataSeq->inputs_count.emplace_back(cols);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskdataSeq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows sum_val_by_rows(taskdataSeq);
  ASSERT_TRUE(sum_val_by_rows.validation());
  sum_val_by_rows.pre_processing();
  sum_val_by_rows.run();
  sum_val_by_rows.post_processing();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_1rand_00x100_matrix) {
  const int rows = 100;
  const int cols = 100;

  // create data
  std::vector<int> in = khokhlov_a_sum_values_by_rows_seq::getRandomMatrix(rows, cols);
  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < cols; j++) {
      tmp_sum += in[i * cols + j];
    }
    expect[i] += tmp_sum;
  }
  std::vector<int> out(rows, 0);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskdataSeq->inputs_count.emplace_back(in.size());
  taskdataSeq->inputs_count.emplace_back(rows);
  taskdataSeq->inputs_count.emplace_back(cols);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskdataSeq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows sum_val_by_rows(taskdataSeq);
  ASSERT_TRUE(sum_val_by_rows.validation());
  sum_val_by_rows.pre_processing();
  sum_val_by_rows.run();
  sum_val_by_rows.post_processing();
  ASSERT_EQ(expect, out);
}

// TEST(khokhlov_a_sum_values_by_rows_seq, )
