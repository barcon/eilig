#ifndef EILIG_THREADED_VECTOR_CPU_HPP_
#define EILIG_THREADED_VECTOR_CPU_HPP_

#include "eilig_threaded.hpp"
#include "eilig_vector.hpp"

namespace eilig
{
    namespace threaded
    {
        VectorCPUPtr CreateVectorCPU();

        class VectorCPU : public IDevice, virtual public std::enable_shared_from_this<VectorCPU>
        {
        public:
            virtual ~VectorCPU() = default;

            static VectorCPUPtr Create();

            Type GetType() const override;

			eilig::Vector& GetVector();

			KernelVectorResize GetKernelResize(NumberRows numberRows);
			KernelVectorCopyScalar GetKernelCopyScalar(Scalar value);
			KernelVectorCopyVector GetKernelCopyVector(const eilig::Vector& vector);
			KernelVectorAddScalar GetKernelAddScalar(Scalar value);
            KernelVectorInitializerList GetKernelVectorInitializerList(const std::initializer_list<Scalar>& values);

        protected:
            VectorCPU() = default;

            const Type type_{ device_vector_cpu };

            eilig::Vector vector_;
		};

        class KernelVectorResize
        {
        public:
            KernelVectorResize(eilig::Vector& vector, NumberRows numberRows) : vector_(vector), numberRows_(numberRows) {};

            ~KernelVectorResize() = default;

            bool operator()()
            {
                vector_.Resize(numberRows_);

				return true;
            }

        private:
            eilig::Vector& vector_;

            NumberRows numberRows_;
        };
        class KernelVectorCopyScalar
        {
        public:
            KernelVectorCopyScalar(eilig::Vector& vector, Scalar value) : vector_(vector), value_(value) {};
            
            ~KernelVectorCopyScalar() = default;

            bool operator()()
            {
                for(Index i = 0; i < vector_.GetRows(); ++i)
                {
                    vector_.data_[i] = value_;
				}

				return true;
            }

        private:
            eilig::Vector& vector_;

            Scalar value_;
        };
        class KernelVectorCopyVector
        {
        public:
            KernelVectorCopyVector(const eilig::Vector& vector) : vector_(vector) {};

            ~KernelVectorCopyVector() = default;

            bool operator()()
            {
                return true;
            }

        private:
            const eilig::Vector& vector_;
        };
        class KernelVectorAddScalar
        {
        public:
            KernelVectorAddScalar(eilig::Vector& vector, Scalar value) : vector_(vector), value_(value) {};

            ~KernelVectorAddScalar() = default;

            bool operator()()
            {
                for (Index i = 0; i < vector_.GetRows(); ++i)
                {
                    vector_.data_[i] += value_;
                }

                return true;
            }

        private:
            eilig::Vector& vector_;

            Scalar value_;
        };
        class KernelVectorInitializerList
        {
        public:
            KernelVectorInitializerList(eilig::Vector& vector, const std::initializer_list<Scalar>& values) : vector_(vector), values_(values) {};

            ~KernelVectorInitializerList() = default;

            bool operator()()
            {
                Index i = 0;
                for (auto& value : values_)
                {
                    vector_.data_[i] = value;
                    ++i;
                }

                return true;
            }

        private:
            eilig::Vector& vector_;

            const std::initializer_list<Scalar>& values_;
        };

    } // namespace threaded
} // namespace eilig

#endif