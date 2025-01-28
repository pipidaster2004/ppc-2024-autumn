#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "seq/shvedova_v_char_freq/include/ops_seq.hpp"

TEST(shvedova_v_char_frequency_seq, test_char_frequency_a_in_abc) {
  std::string input_str = "abcabc";
  char target_char = 'a';
  int expected_frequency = 2;

  std::vector<std::string> in_str(1, input_str);
  std::vector<char> in_char(1, target_char);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_str.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_char.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->inputs_count.emplace_back(in_char.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shvedova_v_char_frequency_seq::CharFrequencyTaskSequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_frequency, out[0]);
}

TEST(shvedova_v_char_frequency_seq, test_char_frequency_b_in_abc) {
  std::string input_str = "abcabc";
  char target_char = 'b';
  int expected_frequency = 2;

  std::vector<std::string> in_str(1, input_str);
  std::vector<char> in_char(1, target_char);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_str.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_char.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->inputs_count.emplace_back(in_char.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shvedova_v_char_frequency_seq::CharFrequencyTaskSequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_frequency, out[0]);
}

TEST(shvedova_v_char_frequency_seq, test_char_frequency_c_in_abc) {
  std::string input_str = "abcabc";
  char target_char = 'c';
  int expected_frequency = 2;

  std::vector<std::string> in_str(1, input_str);
  std::vector<char> in_char(1, target_char);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_str.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_char.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->inputs_count.emplace_back(in_char.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shvedova_v_char_frequency_seq::CharFrequencyTaskSequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_frequency, out[0]);
}

TEST(shvedova_v_char_frequency_seq, test_char_frequency_x_in_abc) {
  std::string input_str = "abcabc";
  char target_char = 'x';
  int expected_frequency = 0;

  std::vector<std::string> in_str(1, input_str);
  std::vector<char> in_char(1, target_char);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_str.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_char.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->inputs_count.emplace_back(in_char.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shvedova_v_char_frequency_seq::CharFrequencyTaskSequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_frequency, out[0]);
}

TEST(shvedova_v_char_frequency_seq, test_char_frequency_a_in_long_string) {
  std::string input_str(1000000, 'a');
  char target_char = 'a';
  int expected_frequency = 1000000;

  std::vector<std::string> in_str(1, input_str);
  std::vector<char> in_char(1, target_char);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_str.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_char.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->inputs_count.emplace_back(in_char.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shvedova_v_char_frequency_seq::CharFrequencyTaskSequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_frequency, out[0]);
}

TEST(shvedova_v_char_frequency_seq, test_char_frequency_in_empty_string) {
  std::string input_str;
  char target_char = 'a';
  int expected_frequency = 0;

  std::vector<std::string> in_str(1, input_str);
  std::vector<char> in_char(1, target_char);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_str.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_char.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->inputs_count.emplace_back(in_char.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shvedova_v_char_frequency_seq::CharFrequencyTaskSequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_frequency, out[0]);
}