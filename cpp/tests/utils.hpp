#ifndef TOPPRA_TEST_UTILS_HPP
#define TOPPRA_TEST_UTILS_HPP
#include <vector>
#include <iostream>
#include <toppra/toppra.hpp>


#define PRINT(X) std::cout << X << std::endl

namespace toppra {

using vvvectors = std::vector<std::vector<value_type> >;
inline Vectors makeVectors(vvvectors v) {
    Vectors ret;
    for (auto vi : v) {
        Vector vi_eigen(vi.size());
        for (std::size_t i = 0; i < vi.size(); i++) vi_eigen(i) = vi[i];
        ret.push_back(vi_eigen);
        }
    return ret;
}

}  // namespace toppra

#endif
