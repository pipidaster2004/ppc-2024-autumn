#include <boost/mpi/timer.hpp>
#include <chrono>
#include <thread>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/khasanyanov_k_average_vector/include/avg_mpi.hpp"

//=========================================sequence=========================================

TEST(khasanyanov_k_average_vector_seq, test_pipeline_run) {
  std::vector<int> global_vec(2212000, 4);
  std::vector<double> average(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_mpi::create_task_data<int, double>(global_vec, average);

  auto testAvgVectorSequence =
      std::make_shared<khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<int, double>>(taskData);

  RUN_TASK(*testAvgVectorSequence);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testAvgVectorSequence);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_NEAR(4, average[0], 1e-5);
}

TEST(khasanyanov_k_average_vector_seq, test_task_run) {
  std::vector<int> global_vec(3050000, 4);
  std::vector<double> average(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_mpi::create_task_data<int, double>(global_vec, average);

  auto testAvgVectorSequence =
      std::make_shared<khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<int, double>>(taskData);

  RUN_TASK(*testAvgVectorSequence);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testAvgVectorSequence);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_NEAR(4, average[0], 1e-5);
}

//=========================================parallel=========================================

TEST(khasanyanov_k_average_vector_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> global_vec;
  std::vector<double> average(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 3050000;
    global_vec = std::vector<double>(count_size_vector, 4);
    taskDataPar = khasanyanov_k_average_vector_mpi::create_task_data<double, double>(global_vec, average);
  }

  auto testMpiTaskParallel =
      std::make_shared<khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<double, double>>(taskDataPar);

  RUN_TASK(*testMpiTaskParallel);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_NEAR(4, average[0], 1e-5);
  }
}

TEST(khasanyanov_k_average_vector_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> global_vec;
  std::vector<double> average(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 3050000;
    global_vec = std::vector<double>(count_size_vector, 4);
    taskDataPar = khasanyanov_k_average_vector_mpi::create_task_data<double, double>(global_vec, average);
  }

  auto testMpiTaskParallel =
      std::make_shared<khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<double, double>>(taskDataPar);

  RUN_TASK(*testMpiTaskParallel);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(4, average[0], 1e-5);
  }
}