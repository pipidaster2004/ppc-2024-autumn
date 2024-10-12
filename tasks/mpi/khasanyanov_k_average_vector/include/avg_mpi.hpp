#ifndef _AVG_MPI_HPP_
#define _AVG_MPI_HPP_

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khasanyanov_k_average_vector_mpi {

std::vector<int> getRandomVector(int size);

template <class In, class Out>
class AvgVectorMPITaskSequential : public ppc::core::Task {
  std::vector<In> input_;
  Out avg = 0.0;

 public:
  explicit AvgVectorMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::pre_processing() {
  internal_order_test();
  input_ = std::vector<In>(taskData->inputs_count[0]);
  auto* tmp = reinterpret_cast<In*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0], std::back_inserter(input_));
  avg = 0.0;
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::run() {
  internal_order_test();
  avg = static_cast<Out>(std::accumulate(input_.begin(), input_.end(), 0.0, std::plus()));
  avg /= static_cast<Out>(taskData->inputs_count[0]);
  // std::this_thread::sleep_for(std::chrono::milliseconds(50));
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::post_processing() {
  internal_order_test();
  reinterpret_cast<Out*>(taskData->outputs[0])[0] = avg;
  return true;
}

}  //  namespace khasanyanov_k_average_vector_mpi

#endif  // !_AVG_MPI_HPP_
