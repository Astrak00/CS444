#include <iostream>
#include <chrono>
#include <vector>
#include <chrono>
#include <iomanip>

#define TIMES_COMPARAISON 100'000'000

template <typename Func>
void measure_time(const std::string &operation, Func func, int a, int comp, std::vector<bool> &solution)
{
  bool c;
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < TIMES_COMPARAISON; ++i)
  {
    c = func(a, comp);
    solution.push_back(c);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = end - start;
  std::cout << operation << " has taken: " << std::fixed << std::setprecision(10) << elapsed.count() / 1'000 << " nanoseconds, resulting in " << c << std::endl;
  solution.clear();
}

int main(int argc, char *argv[])
{
  std::cout << "paso por aqui \n";

  if (argc != 4)
  {
    std::cerr << "Usage: " << argv[0] << " <a> <b>" << std::endl;
    return 1;
  }
  std::cout << "paso por aqui 2 \n";

  std::vector<bool> solution;

  const int a = std::stoi(argv[1]);
  const int b = std::stoi(argv[2]);
  bool c;
  solution.reserve(TIMES_COMPARAISON);

  std::vector<std::pair<std::string, std::function<bool(int, int)>>> operations = {
      {"!=", [](int a, int b) { return a != b; }},
      {"==", [](int a, int b) { return a == b; }},
      {"< ", [](int a, int b) { return a < b; }},
      {"> ", [](int a, int b) { return a > b; }},
      {"<=", [](int a, int b) { return a <= b; }},
      {">=", [](int a, int b) { return a >= b; }},
      {"&&", [](int a, int b) { return a && b; }},
      {"||", [](int a, int b) { return a || b; }},
      {"! ", [](int a, int b) { return !a; }},
      {"~ ", [](int a, int b) { return ~a; }},
      {"++", [](int a, int b) { return ++a; }},
      {"--", [](int a, int b) { return --a; }},
      {"+ ", [](int a, int b) { return a + b; }}};

  for (const auto &op : operations)
  {
    measure_time(op.first, op.second, a, b, solution);
  }
}