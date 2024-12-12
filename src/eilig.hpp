#ifndef EILIG_HPP_
#define EILIG_HPP_

#include "eilig_types.hpp"
#include "eilig_status.hpp"
#include "eilig_vector.hpp"
#include "eilig_matrix.hpp"
#include "eilig_matrix_ellpack.hpp"

#ifdef EILIG_ENABLE_OPENCL
#include "eilig_opencl_entry_proxy.hpp"
#include "eilig_opencl_kernels.hpp"
#include "eilig_opencl_vector.hpp"
#include "eilig_opencl_matrix_ellpack.hpp"
#endif // EILIG_ENABLE_OPENCL

#include "eilig_routines.hpp"
#include "eilig_transform.hpp"

#endif /* EILIG_HPP_ */