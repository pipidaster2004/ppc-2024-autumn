#ifndef _AVG_MPI_HPP_
#define _AVG_MPI_HPP_

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khasanyanov_k_average_vector_mpi {

std::vector<int> getRandomVector(int size);

//=========================================sequential=========================================

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
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

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
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::run() {
  internal_order_test();
  avg = static_cast<Out>(std::accumulate(input_.begin(), input_.end(), 0.0, std::plus()));
  avg /= static_cast<Out>(taskData->inputs_count[0]);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::post_processing() {
  internal_order_test();
  reinterpret_cast<Out*>(taskData->outputs[0])[0] = avg;
  return true;
}

//=========================================parallel=========================================

namespace mpi = boost::mpi;
template <class In, class Out>
class AvgVectorMPITaskParallel : public ppc::core::Task {
  std::vector<In> input_, local_input_;
  Out avg = 0.0;
  mpi::communicator world;

 public:
  explicit AvgVectorMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<In, Out>::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<In, Out>::pre_processing() {
  internal_order_test();
  size_t part;
  if (world.rank() == 0) {
    part = taskData->inputs_count[0] / world.size();
  }
  mpi::broadcast(world, part, 0);

  if (world.rank() == 0) {
    input_ = std::vector<In>(taskData->inputs_count[0]);
    auto* tmp = reinterpret_cast<In*>(taskData->inputs[0]);
    for (size_t i = 0; i < taskData->inputs_count[0]; ++i) {
      input_[i] = tmp[i];
    }
    // std::copy(tmp, tmp + taskData->inputs_count[0], std::back_inserter(input_));
    for (int num = 1; num < world.size(); ++num) {
      world.send(num, 0, input_.data() + num * part, part);
    }
  }

  local_input_ = std::vector<In>(part);
  if (world.rank() == 0) {
    local_input_ = std::vector<In>(input_.begin(), input_.begin() + part);
  } else {
    world.recv(0, 0, local_input_.data(), part);
  }

  avg = 0.0;
  return true;
}

// using namespace std::chrono_literals;
template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<In, Out>::run() {
  internal_order_test();
  Out local_avg{};
  local_avg = static_cast<Out>(std::accumulate(local_input_.begin(), local_input_.end(), 0.0, std::plus()));
  local_avg /= static_cast<Out>(local_input_.size());
  mpi::reduce(world, local_avg / world.size(), avg, std::plus(), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<In, Out>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<Out*>(taskData->outputs[0])[0] = avg;
  }
  return true;
}

}  //  namespace khasanyanov_k_average_vector_mpi

#endif  // !_AVG_MPI_HPP_
