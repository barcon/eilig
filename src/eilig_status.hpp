#ifndef EILIG_STATUS_HPP_
#define EILIG_STATUS_HPP_

#include "eilig_types.hpp"
#include <map>

namespace eilig
{

	static const Status EILIG_SUCCESS = 0;
	static const Status EILIG_NOT_CONVERGED = -1;
	static const Status EILIG_INVALID_TOLERANCE = -2;
	static const Status EILIG_INVALID_FILE = -3;

	static const std::map<Status, String> messages =
	{
		{  0, "EILIG_SUCCESS"},
		{ -1, "EILIG_NOT_CONVERGED"},
		{ -2, "EILIG_INVALID_TOLERANCE"},
		{ -3, "EILIG_INVALID_FILE"}
	};

}

#endif