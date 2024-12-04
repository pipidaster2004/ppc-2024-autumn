#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/khokhlov_a_iterative_seidel_method/include/ops_mpi_khokhlov.hpp"

TEST(khokhlov_a_iterative_seidel_method_mpi, test_empty_matrix) {
  boost::mpi::communicator world;
  const int n = 0;
  const int maxiter = 10;

  // create data
  std::vector<double> A = {};
  std::vector<double> b = {};
  std::vector<double> expect = {};
  std::vector<double> result = {};

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  }

  // crate task
  khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi seidel_method_mpi(taskdataMpi);
  ASSERT_FALSE(seidel_method_mpi.validation());

}

TEST(khokhlov_a_iterative_seidel_method_mpi, test_invalid_iter) {
  boost::mpi::communicator world;
  const int n = 2;
  const int maxiter = 0;

  // create data
  std::vector<double> A = {1, 2, 3, 4};
  std::vector<double> b = {1, 2};
  std::vector<double> expect = {};
  std::vector<double> result = {};

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  }

  // crate task
  khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi seidel_method_mpi(taskdataMpi);
  ASSERT_FALSE(seidel_method_mpi.validation());
}

TEST(khokhlov_a_iterative_seidel_method_mpi, test_const_matrix) {
  boost::mpi::communicator world;
  const int n = 300;
  const int maxiter = 1000;

  // Create data
  std::vector<double> A(n * n, 0.0);
  std::vector<double> b(n, 0.0);
  std::vector<double> result(n, 0.0);
  khokhlov_a_iterative_seidel_method_mpi::getRandomSLAU(A, b, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  };
  khokhlov_a_iterative_seidel_method_mpi::seidel_method_seq seidel_method_mpi(taskdataMpi);
  ASSERT_TRUE(seidel_method_mpi.validation());
  std::cout << world.rank() << std::endl;
  seidel_method_mpi.pre_processing();
  std::cout << world.rank() << std::endl;
  seidel_method_mpi.run();
  std::cout << world.rank() << std::endl;
  seidel_method_mpi.post_processing();
  std::cout << world.rank() << std::endl;

  ASSERT_EQ(result.size(), b.size());

  if (world.rank() == 0) {
    std::vector<double> result_seq(n, 0.0);
    // create task data
    std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
    taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataSeq->inputs_count.emplace_back(n);
    taskdataSeq->inputs_count.emplace_back(maxiter);
    taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
    taskdataSeq->outputs_count.emplace_back(result_seq.size());

    // crate task
    khokhlov_a_iterative_seidel_method_mpi::seidel_method_seq seidel_method_seq(taskdataSeq);
    ASSERT_TRUE(seidel_method_seq.validation());
    seidel_method_seq.pre_processing();
    seidel_method_seq.run();
    seidel_method_seq.post_processing();
    if (world.rank() == 0) {
      for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], result_seq[i], 1e-1);
      }
    }
  }
}