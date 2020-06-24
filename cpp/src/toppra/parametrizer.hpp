#ifndef TOPPRA_PARAMETRIZER_HPP
#define TOPPRA_PARAMETRIZER_HPP

#include <toppra/geometric_path.hpp>

namespace toppra {

/**
 * \brief Abstract interface for output trajectory parametrizers.
 * 
 * A parametrizer provides exactly the same functionality as a
 * geometric path object.
 */
class Parametrizer: public GeometricPath {
};
};

#endif
