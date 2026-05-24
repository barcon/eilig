#include "eilig_threaded.hpp"

namespace eilig
{
    namespace threaded
    {
        NumberProcessors numberProcessors = std::thread::hardware_concurrency() > 1 ? std::thread::hardware_concurrency() : 1;
        ThreadPool threadPool(numberProcessors);
    }
} /* namespace eilig */