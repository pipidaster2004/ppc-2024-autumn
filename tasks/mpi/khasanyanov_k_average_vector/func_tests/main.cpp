#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"
#include "gtest/gtest.h"
#include "mpi/khasanyanov_k_average_vector/include/avg_mpi.hpp"

//=========================================sequence=========================================

TEST(khasanyanov_k_average_vector_seq, test_int) {
  std::vector<int32_t> in(3333, 77);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_mpi::create_task_data<int32_t, double>(in, out);

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<int32_t, double> testTask(taskData);

  RUN_TASK(testTask);
  EXPECT_NEAR(out[0], 77, 1e-5);

  // ASSERT_TRUE(testTask.validation());
  // testTask.pre_processing();
  // testTask.run();
  // testTask.post_processing();
  // EXPECT_NEAR(out[0], 77, 1e-5);
}

TEST(khasanyanov_k_average_vector_seq, test_double) {
  std::vector<double> in(3333, 7.7);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_mpi::create_task_data<double, double>(in, out);

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<double, double> testTask(taskData);

  RUN_TASK(testTask);
  EXPECT_NEAR(out[0], 7.7, 1e-5);
}

TEST(khasanyanov_k_average_vector_seq, test_float) {
  std::vector<float> in(3333, 7.7);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_mpi::create_task_data<float, double>(in, out);

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<float, double> testTask(taskData);

  RUN_TASK(testTask);
  EXPECT_NEAR(out[0], 7.7, 1e-5);
}

TEST(khasanyanov_k_average_vector_seq, test_random) {
  std::vector<double> in = khasanyanov_k_average_vector_mpi::get_random_vector<double>(15);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_mpi::create_task_data<double, double>(in, out);

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<double, double> testTask(taskData);
  RUN_TASK(testTask);

  double expect_res = std::accumulate(in.begin(), in.end(), 0.0, std::plus()) / in.size();
  EXPECT_NEAR(out[0], expect_res, 1e-5);
}

TEST(khasanyanov_k_average_vector_seq, test_uint) {
  std::vector<std::uint8_t> in(1200, 3);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_mpi::create_task_data<std::uint8_t, double>(in, out);

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<std::uint8_t, double> testTask(taskData);
  RUN_TASK(testTask);
  EXPECT_NEAR(out[0], 3, 1e-5);
}

//=========================================parallel=========================================

namespace mpi = boost::mpi;

TEST(khasanyanov_k_average_vector_mpi, test_float) {
  mpi::communicator world;
  std::vector<float> in(1234, 3.3);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = khasanyanov_k_average_vector_mpi::create_task_data<float, double>(in, out);
  }

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<float, double> testTask(taskData);

  RUN_TASK(testTask);

  if (world.rank() == 0) {
    std::vector<double> seq_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq =
        khasanyanov_k_average_vector_mpi::create_task_data<float, double>(in, seq_out);

    khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<float, double> testMpiTaskSequential(taskDataSeq);

    RUN_TASK(testMpiTaskSequential);

    EXPECT_NEAR(seq_out[0], out[0], 1e-5);
  }
}

TEST(khasanyanov_k_average_vector_mpi, test_int) {
  mpi::communicator world;
  std::vector<int> in(1234, 3);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = khasanyanov_k_average_vector_mpi::create_task_data<int, double>(in, out);
  }

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<int, double> testTask(taskData);

  RUN_TASK(testTask);

  if (world.rank() == 0) {
    std::vector<double> seq_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq =
        khasanyanov_k_average_vector_mpi::create_task_data<int, double>(in, seq_out);

    khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<int, double> testMpiTaskSequential(taskDataSeq);

    RUN_TASK(testMpiTaskSequential);

    EXPECT_NEAR(seq_out[0], out[0], 1e-5);
  }
}

TEST(khasanyanov_k_average_vector_mpi, test_uint8) {
  mpi::communicator world;
  std::vector<std::uint8_t> in(1234, 3);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = khasanyanov_k_average_vector_mpi::create_task_data<std::uint8_t, double>(in, out);
  }

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<std::uint8_t, double> testTask(taskData);

  RUN_TASK(testTask);

  if (world.rank() == 0) {
    std::vector<double> seq_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq =
        khasanyanov_k_average_vector_mpi::create_task_data<std::uint8_t, double>(in, seq_out);

    khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<std::uint8_t, double> testMpiTaskSequential(
        taskDataSeq);

    RUN_TASK(testMpiTaskSequential);

    EXPECT_NEAR(seq_out[0], out[0], 1e-5);
  }
}
