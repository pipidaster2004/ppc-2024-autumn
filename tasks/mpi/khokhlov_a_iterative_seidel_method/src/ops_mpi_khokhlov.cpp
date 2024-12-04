#include "mpi/khokhlov_a_iterative_seidel_method/include/ops_mpi_khokhlov.hpp"

bool khokhlov_a_iterative_seidel_method_mpi::seidel_method_seq::pre_processing() {
  internal_order_test();
  // init matrix
  A = std::vector<double>(taskData->inputs_count[0] * taskData->inputs_count[0]);
  auto tmp = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0] * taskData->inputs_count[0], A.begin());

  // init vector
  b = std::vector<double>(taskData->inputs_count[0]);
  auto tmp1 = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(tmp1, tmp1 + taskData->inputs_count[0], b.begin());

  // init world.size()s
  n = taskData->inputs_count[0];

  // init maxIterations
  maxIterations = taskData->inputs_count[1];

  // Init value for output
  result = std::vector<double>(taskData->inputs_count[0], 0);
  return true;
}

bool khokhlov_a_iterative_seidel_method_mpi::seidel_method_seq::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0);
}

bool khokhlov_a_iterative_seidel_method_mpi::seidel_method_seq::run() {
  internal_order_test();
  std::vector<double> x(n, 1.0);
  std::vector<double> prevX(n, 1.0);

  for (int iter = 0; iter < maxIterations; ++iter) {
    for (int i = 0; i < n; ++i) {
      double sum = 0;
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          sum += (A[i * n + j] * x[j]);
        }
      }
      if (A[i * n + i] != 0) {
        x[i] = (b[i] - sum) / A[i * n + i];
      } else {
        x[i] = 0.0;
      }
    }
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
      norm += std::pow(x[i] - prevX[i], 2);
    }
    norm = std::sqrt(norm);

    if (norm < EPSILON) {
      break;
    }
    prevX = x;
  }
  result = x;
  return true;
}

bool khokhlov_a_iterative_seidel_method_mpi::seidel_method_seq::post_processing() {
  internal_order_test();
  for (int i = 0; i < n; i++) reinterpret_cast<double*>(taskData->outputs[0])[i] = result[i];
  return true;
}

bool khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // init matrix
    A = std::vector<double>(taskData->inputs_count[0] * taskData->inputs_count[0]);
    auto tmp = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(tmp, tmp + taskData->inputs_count[0] * taskData->inputs_count[0], A.begin());

    // init vector
    b = std::vector<double>(taskData->inputs_count[0]);
    auto tmp1 = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(tmp1, tmp1 + taskData->inputs_count[0], b.begin());

    // init world.size()s
    n = taskData->inputs_count[0];

    // init maxIterations
    maxIterations = taskData->inputs_count[1];

    // Init value for output
    x = std::vector<double>(taskData->inputs_count[0], 0);
  }
  return true;
}

bool khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0);
  }
  return true;
}

bool khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi::run() {
  internal_order_test();
  boost::mpi::broadcast(world, n, 0);
  boost::mpi::broadcast(world, maxIterations, 0);
  int delta = n / world.size();
  int last_rows = n % world.size();

  int local_n = (world.rank() == world.size() - 1) ? delta + last_rows : delta;

  local_A.resize(local_n * n);
  local_b.resize(local_n);
  local_x.resize(local_n);
  prevX.resize(n, 0.0);

  std::vector<int> send_counts_A(world.size());
  std::vector<int> send_counts_b(world.size());
  std::vector<int> displs_A(world.size());
  std::vector<int> displs_b(world.size());

  for (int i = 0; i < world.size(); ++i) {
    send_counts_A[i] = (i == world.size() - 1) ? delta + last_rows : delta;
    send_counts_A[i] *= n;
    displs_A[i] = (i > 0) ? displs_A[i - 1] + send_counts_A[i - 1] : 0;
    send_counts_b[i] = (i == world.size() - 1) ? delta + last_rows : delta;
    displs_b[i] = (i > 0) ? displs_b[i - 1] + send_counts_b[i - 1] : 0;
  }
  // if (world.rank() == 0) {
  boost::mpi::scatterv(world, A.data(), send_counts_A, /*displs_A,*/ local_A.data(), /*send_counts_A[world.rank()],*/ 0);
  boost::mpi::scatterv(world, b.data(), send_counts_b, /*displs_b,*/ local_b.data(), /*send_counts_b[world.rank()],*/ 0);
  //} else {
  //  boost::mpi::scatterv(world, local_A.data(), send_counts_A[world.rank()], 0);
  //  boost::mpi::scatterv(world, local_b.data(), send_counts_b[world.rank()] / n, 0);
  //}

  for (int iter = 0; iter < maxIterations; ++iter) {
    for (int i = 0; i < local_n; ++i) {
      double sum = 0;
      int global_i = displs_b[world.rank()] + i;
      for (int j = 0; j < n; ++j) {
        if (j != global_i) {
          sum += local_A[i * n + j] * x[j];
        }
      }
      if (local_A[i * n + i] != 0) {
        local_x[i] = (local_b[i] - sum) / local_A[i * n + global_i];
      } else {
        local_x[i] = 0;
      }
    }

    // boost::mpi::all_gather(world, local_x.data(), local_n, x.data());

    double local_norm = 0.0;
    for (int i = 0; i < local_n; ++i) {
      int global_i = displs_b[world.rank()] + i;
      local_norm += std::pow(x[global_i] - prevX[global_i], 2);
    }

    double global_norm = 0.0;
    boost::mpi::all_reduce(world, local_norm, global_norm, std::plus<double>());
    global_norm = std::sqrt(global_norm);

    boost::mpi::gatherv(world, local_x.data(), local_n, x.data(), send_counts_b, /*displs_b,*/ 0);

    if (world.rank() == 0) {
      prevX = x;
    }
    boost::mpi::broadcast(world, prevX.data(), n, 0);

    if (global_norm < EPSILON) {
      break;
    }
  }
  return true;
}

bool khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < n; i++) reinterpret_cast<double*>(taskData->outputs[0])[i] = x[i];
  }
  return true;
}

void khokhlov_a_iterative_seidel_method_mpi::getRandomSLAU(std::vector<double>& A, std::vector<double>& b, int N) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < N; ++i) {
    double rowSum = 0.0;
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        A[i * N + j] = rand() % 10 - 5;
        rowSum += std::abs(A[i * N + j]);
      }
    }
    A[i * N + i] = rowSum + (rand() % 5 + 1);
    b[i] = rand() % 20 - 10;
  }
}