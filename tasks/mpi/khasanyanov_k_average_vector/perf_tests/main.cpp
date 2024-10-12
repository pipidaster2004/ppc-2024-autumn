#include <boost/mpi/timer.hpp>
#include <chrono>
#include <thread>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/khasanyanov_k_average_vector/include/avg_mpi.hpp"

TEST(khasanyanov_k_average_vector_seq, test_pipeline_run) {
  std::vector<int> global_vec(2212000, 4);
  std::vector<double> average(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  taskData->inputs_count.emplace_back(global_vec.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(average.data()));
  taskData->outputs_count.emplace_back(average.size());

  auto testAvgVectorSequence =
      std::make_shared<khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<int, double>>(taskData);
  ASSERT_TRUE(testAvgVectorSequence->validation());
  testAvgVectorSequence->pre_processing();
  testAvgVectorSequence->run();
  testAvgVectorSequence->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testAvgVectorSequence);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(4, average[0]);
}

TEST(khasanyanov_k_average_vector_seq, test_task_run) {
  std::vector<int> global_vec(3050000, 4);
  std::vector<double> average(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  taskData->inputs_count.emplace_back(global_vec.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(average.data()));
  taskData->outputs_count.emplace_back(average.size());

  auto testAvgVectorSequence =
      std::make_shared<khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<int, double>>(taskData);
  ASSERT_TRUE(testAvgVectorSequence->validation());
  testAvgVectorSequence->pre_processing();
  testAvgVectorSequence->run();
  testAvgVectorSequence->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testAvgVectorSequence);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(4, average[0]);
}
