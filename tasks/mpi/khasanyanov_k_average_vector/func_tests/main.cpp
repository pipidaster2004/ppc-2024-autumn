#include <iostream>
#include <vector>

#include "mpi/khasanyanov_k_average_vector/include/avg_mpi.hpp"


TEST(khasanyanov_k_average_vector_seq, test_int) {
  std::vector<int32_t> in(3333, 77);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<int32_t, double> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_TRUE(isValid);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  std::cout << out[0] << std::endl;
  EXPECT_NEAR(out[0], 77.0, 1e-5);
}

TEST(khasanyanov_k_average_vector_seq, test_double) {
  std::vector<double> in(3333, 7.7);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<double, double> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_TRUE(isValid);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  EXPECT_NEAR(out[0], 7.7, 1e-5);
}

TEST(khasanyanov_k_average_vector_seq, test_float) {
  std::vector<float> in(3333, 7.7);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<float, double> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_TRUE(isValid);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  EXPECT_NEAR(out[0], 7.7, 1e-5);
}

TEST(khasanyanov_k_average_vector_seq, test_uint) {
  std::vector<std::uint8_t> in(1200, 3);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<std::uint8_t, double> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_TRUE(isValid);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  EXPECT_NEAR(out[0], 3, 1e-5);
}
