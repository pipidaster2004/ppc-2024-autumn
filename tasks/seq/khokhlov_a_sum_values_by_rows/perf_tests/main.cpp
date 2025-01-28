#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/khokhlov_a_sum_values_by_rows/include/ops_sec.hpp"

TEST(khokhlov_a_sum_values_by_rows_seq, test_pipline_run_seq) {
  const int rows = 5000;
  const int cols = 5000;

  // create data
  std::vector<int> in(cols * rows, 0);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) in[i] += i * cols + j;
  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < cols; j++) tmp_sum += in[i * cols + j];
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
  auto testTaskSeq = std::make_shared<khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows>(taskdataSeq);

  // create perf attrib
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_task_run_seq) {
  const int rows = 5000;
  const int cols = 5000;

  // create data
  std::vector<int> in(cols * rows, 0);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) in[i] += i * cols + j;
  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < cols; j++) tmp_sum += in[i * cols + j];
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
  auto testTaskSeq = std::make_shared<khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows>(taskdataSeq);

  // Create Task
  auto testTaskSequential = std::make_shared<khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows>(taskdataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expect, out);
}