#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "coalsack/llm/chat_template.h"

using namespace coalsack;
using json = nlohmann::json;

int main(int argc, char** argv) {
  std::cout << "Testing Chat Template\n";
  std::cout << "=====================\n\n";

  std::string json_path = argc > 1 ? argv[1] : "test/chat_template_test_data.json";

  std::ifstream json_file(json_path);
  if (!json_file) {
    std::cerr << "ERROR: Cannot open " << json_path << "\n";
    return 1;
  }

  json test_data;
  try {
    json_file >> test_data;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: Failed to parse JSON: " << e.what() << "\n";
    return 1;
  }

  chat_template tpl;

  std::cout << "Test 1: System + User message\n";
  tpl.add_system(test_data["test1"]["system"]);
  tpl.add_user(test_data["test1"]["user"]);

  std::string prompt = tpl.build_prompt();
  std::cout << "  Generated prompt:\n";
  std::cout << "  " << prompt << "\n\n";

  std::string expected_start = test_data["test1"]["expected_system"];
  if (prompt.find(expected_start) != std::string::npos) {
    std::cout << "  ✓ System message format correct\n";
  } else {
    std::cerr << "  ERROR: System message format incorrect\n";
    return 1;
  }

  std::string expected_user = test_data["test1"]["expected_user"];
  if (prompt.find(expected_user) != std::string::npos) {
    std::cout << "  ✓ User message format correct\n";
  } else {
    std::cerr << "  ERROR: User message format incorrect\n";
    return 1;
  }

  std::string expected_suffix = test_data["test1"]["expected_suffix"];
  if (prompt.find(expected_suffix) != std::string::npos) {
    std::cout << "  ✓ Assistant prompt suffix correct\n";
  } else {
    std::cerr << "  ERROR: Assistant prompt suffix incorrect\n";
    return 1;
  }

  std::cout << "\nTest 2: Response extraction\n";
  std::string generated = test_data["test2"]["generated"];
  std::string extracted = tpl.extract_final_response(generated);
  std::cout << "  Input: " << generated << "\n";
  std::cout << "  Extracted: \"" << extracted << "\"\n";

  std::string expected_extracted = test_data["test2"]["expected_extracted"];
  if (extracted == expected_extracted) {
    std::cout << "  ✓ Response extraction correct\n";
  } else {
    std::cerr << "  ERROR: Expected '" << expected_extracted << "', got '" << extracted << "'\n";
    return 1;
  }

  std::cout << "\nTest 3: Multi-turn conversation\n";
  tpl.clear();

  tpl.add_system(test_data["test3"]["system"]);
  tpl.add_user(test_data["test3"]["user1"]);
  tpl.add_assistant(test_data["test3"]["assistant1"]);
  tpl.add_user(test_data["test3"]["user2"]);

  std::string multi_prompt = tpl.build_prompt();
  std::cout << "  Message count: " << tpl.message_count() << "\n";
  std::cout << "  Generated prompt:\n";
  std::cout << "  " << multi_prompt << "\n\n";

  int expected_count = test_data["test3"]["expected_message_count"];
  if (tpl.message_count() == static_cast<size_t>(expected_count)) {
    std::cout << "  ✓ Message count correct\n";
  } else {
    std::cerr << "  ERROR: Expected " << expected_count << " messages, got " << tpl.message_count()
              << "\n";
    return 1;
  }

  std::string expected_system3 = test_data["test3"]["expected_system"];

  if (multi_prompt.find(expected_system3) != std::string::npos) {
    std::cout << "  ✓ System message preserved\n";
  } else {
    std::cerr << "  ERROR: System message not found\n";
    return 1;
  }

  std::string expected_assistant = test_data["test3"]["expected_assistant"];

  if (multi_prompt.find(expected_assistant) != std::string::npos) {
    std::cout << "  ✓ Assistant message with channel correct\n";
  } else {
    std::cerr << "  ERROR: Assistant message format incorrect\n";
    return 1;
  }

  std::cout << "\n✓ All tests passed!\n";

  return 0;
}
