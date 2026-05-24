#ifndef EILIG_THREADED_HPP_
#define EILIG_THREADED_HPP_

#include "eilig_types.hpp"
#include "BS_thread_pool.hpp"

#include <chrono>
#include <future>

using ThreadPool = BS::thread_pool<BS::tp::pause>;

namespace eilig
{
    namespace threaded
    {
		static const Type device_vector_cpu = 1;
		static const Type device_vector_gpu = 2;

		class IDevice
		{
		public:
			virtual ~IDevice() = default;

			virtual Type GetType() const = 0;
		};

		template<typename T>
		bool IsReady(std::future<T> const& f)
		{
			return f.wait_for(std::chrono::duration<Scalar>::zero()) == std::future_status::ready;
		}

		using NumberDevices = Number;
    }
} /* namespace eilig */

#endif /* EILIG_THREADED_HPP_ */