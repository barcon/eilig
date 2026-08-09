#ifndef EILIG_THREADED_HPP_
#define EILIG_THREADED_HPP_

#include "eilig_types.hpp"

#include "eilig_vector.hpp"
#include "eilig_matrix_ellpack.hpp"

#include "eilig_opencl_vector.hpp"
#include "eilig_opencl_matrix_ellpack.hpp"

#include "BS_thread_pool.hpp"
#include <chrono>
#include <future>
#include <deque>

namespace eilig
{
    namespace threaded
    {
		struct Device {
			Tag deviceTag{ 0 };
			Accelerator accelerator{ Accelerator::cpu };
			opencl::KernelsPtr kernels{ nullptr };
		};

		struct Offset {
			Index block{ 0 };
			Index row{ 0 };
		};

		struct Block {
			bool isUsed{ false };

			Index index{ 0 };
			Device device{};
			IKernelPtr kernel{ nullptr };

			Index row{ 0 };	
			NumberRows numberRows{ 0 };
		};

		template<typename T>
		struct TaskBlock
		{
			T value{};

			const Block& block;
		};

		using NumberBlocks = Number;
		using NumberProcessors = Number;
		using Blocks = std::vector<Block>;
		using Devices = std::vector<Device>;
		using ThreadPool = BS::thread_pool<BS::tp::pause>;
		
		using TaskQueueBool = std::deque<std::future<bool>>;
		using TaskQueueScalar = std::deque<std::future<Scalar>>;
		using TaskQueueVector = std::deque<std::future<TaskBlock<eilig::Vector>>>;
		using TaskQueueVectorCL = std::deque<std::future<TaskBlock<eilig::opencl::Vector>>>;
		using TaskQueueEllpack = std::deque<std::future<TaskBlock<eilig::Ellpack>>>;
		using TaskQueueEllpackCL = std::deque<std::future<TaskBlock<eilig::opencl::Ellpack>>>;

		void	AdjustBlock(Block& block, NumberRows numberRows, Number numberDevices, Index index);
		Offset	GetOffset(const Blocks& blocks, Index row);
		Devices	GetDevices(const Blocks& blocks);

		void	WaitForAll(TaskQueueBool& queue);
		void	WaitForAll_Ellpack_Vector_Multiplication(TaskQueueVector& queue, TaskQueueVectorCL& queueCL, Vector& result);
		Scalar	WaitForAll_Ellpack_Sum(TaskQueueScalar& queue);
		Ellpack	WaitForAll_Ellpack_Diagonal(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL);
		Ellpack	WaitForAll_Ellpack_Diagonal_Scale(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL);
		Vector	WaitForAll_Ellpack_Diagonal_Vector(TaskQueueVector& queue, TaskQueueVectorCL& queueCL);
		Ellpack	WaitForAll_Ellpack_Lower_With_Diagonal(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL);
		Ellpack	WaitForAll_Ellpack_Lower_Without_Diagonal(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL);
		Ellpack	WaitForAll_Ellpack_Upper_With_Diagonal(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL);
		Ellpack	WaitForAll_Ellpack_Upper_Without_Diagonal(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL);

		template<typename T>
		bool IsReady(std::future<T> const& f)
		{
			return f.wait_for(std::chrono::duration<Scalar>::zero()) == std::future_status::ready;
		}

		class IKernel
		{
		public:
			virtual ~IKernel() = default;
		};

		template<typename T>
		class KernelVectorResize
        {
        public:
            KernelVectorResize(T& vector, NumberRows numberRows) : vector_(vector), numberRows_(numberRows) {};

            ~KernelVectorResize() = default;

            bool operator()()
            {
                vector_.Resize(numberRows_);

				return true;
            }

        private:
            T& vector_;

            NumberRows numberRows_;
        };
	
		template<typename T>
		class KernelVectorInitializerList
		{
		public:
			KernelVectorInitializerList(T& vector, const std::initializer_list<Scalar>& value) : vector_(vector), value_(value) {};

			~KernelVectorInitializerList() = default;

			bool operator()()
			{
				vector_.Equal(value_);

				return true;
			}

		private:
			T& vector_;

			const std::initializer_list<Scalar>& value_;
		};

		template<typename T>
		class KernelVectorCopyScalar
		{
		public:
			KernelVectorCopyScalar(T& vector, Scalar value) : vector_(vector), value_(value) {};

			~KernelVectorCopyScalar() = default;

			bool operator()()
			{
				vector_.Equal(value_);

				return true;
			}

		private:
			T& vector_;

			Scalar value_;
		};

		template<typename T>
		class KernelVectorCopyVector
		{
		public:
			KernelVectorCopyVector(T& vector, const eilig::Vector& value) : vector_(vector), value_(value) {};

			~KernelVectorCopyVector() = default;

			bool operator()()
			{
				vector_.Equal(value_);

				return true;
			}

		private:
			T& vector_;

			const eilig::Vector& value_;
		};

		template<typename T>
		class KernelVectorCopyVector2
		{
		public:
			KernelVectorCopyVector2(T& vector, const T& value) : vector_(vector), value_(value) {};

			~KernelVectorCopyVector2() = default;

			bool operator()()
			{
				vector_.Equal(value_);

				return true;
			}

		private:
			T& vector_;

			const T& value_;
		};

		template<typename T>
		class KernelVectorAddScalar
		{
		public:
			KernelVectorAddScalar(T& vector, Scalar value) : vector_(vector), value_(value) {};

			~KernelVectorAddScalar() = default;

			bool operator()()
			{
				vector_.Add(value_);

				return true;
			}

		private:
			T& vector_;

			Scalar value_;
		};

		template<typename T>
		class KernelVectorAddVector
		{
		public:
			KernelVectorAddVector(T& vector, const T& value) : vector_(vector), value_(value) {};

			~KernelVectorAddVector() = default;

			bool operator()()
			{
				vector_.Add(value_);

				return true;
			}

		private:
			T& vector_;

			const T& value_;
		};

		template<typename T>
		class KernelVectorSubScalar
		{
		public:
			KernelVectorSubScalar(T& vector, Scalar value) : vector_(vector), value_(value) {};

			~KernelVectorSubScalar() = default;

			bool operator()()
			{
				vector_.Sub(value_);
				return true;
			}

		private:
			T& vector_;

			Scalar value_;
		};

		template<typename T>
		class KernelVectorSubVector
		{
		public:
			KernelVectorSubVector(T& vector, const T& value) : vector_(vector), value_(value) {};

			~KernelVectorSubVector() = default;

			bool operator()()
			{
				vector_.Sub(value_);
				return true;
			}

		private:
			T& vector_;

			const T& value_;
		};

		template<typename T>
		class KernelVectorMulScalar
		{
		public:
			KernelVectorMulScalar(T& vector, Scalar value) : vector_(vector), value_(value) {};

			~KernelVectorMulScalar() = default;

			bool operator()()
			{
				vector_.Mul(value_);
				return true;
			}

		private:
			T& vector_;

			Scalar value_;
		};

		template<typename T>
		class KernelVectorSwapRows
		{
		public:
			KernelVectorSwapRows(T& vector, Index row1, Index row2) : vector_(vector), row1_(row1), row2_(row2) {};

			~KernelVectorSwapRows() = default;

			bool operator()()
			{
				vector_.SwapRows(row1_, row2_);
				return true;
			}

		private:
			T& vector_;

			Index row1_;
			Index row2_;
		};

		template<typename T>
		class KernelVectorRegion
		{
		public:
			KernelVectorRegion(T& vector, const T& value, Index row1, Index row2) : vector_(vector), value_(value), row1_(row1), row2_(row2) {};

			~KernelVectorRegion() = default;

			bool operator()()
			{
				vector_ = value_.Region(row1_, row2_);
				
				return true;
			}

		private:
			T& vector_;
			const T& value_;

			Index row1_;
			Index row2_;
		};

		template<typename T>
		class KernelVectorReplace1
		{
		public:
			KernelVectorReplace1(T& vector, const T& value, Index row1) : vector_(vector), value_(value), row1_(row1) {};

			~KernelVectorReplace1() = default;

			bool operator()()
			{
				vector_.Replace(row1_, value_);

				return true;
			}

		private:
			T& vector_;
			const T& value_;

			Index row1_;
		};

		template<typename T>
		class KernelVectorReplace2
		{
		public:
			KernelVectorReplace2(T& vector, const eilig::Vector& value, Index row1) : vector_(vector), value_(value), row1_(row1) {};

			~KernelVectorReplace2() = default;

			bool operator()()
			{
				vector_.Replace(row1_, value_);

				return true;
			}

		private:
			T& vector_;
			const eilig::Vector& value_;

			Index row1_;
		};

		template<typename T>
		class KernelVectorSetValue
		{
		public:
			KernelVectorSetValue(T& vector, const eilig::Vector& value) : vector_(vector), value_(value) {};

			~KernelVectorSetValue() = default;

			bool operator()()
			{
				vector_.SetValue(value_);

				return true;
			}

		private:
			T& vector_;
			const eilig::Vector& value_;
		};

		template<typename T>
		class KernelVectorSetValue2
		{
		public:
			KernelVectorSetValue2(T& vector, const std::initializer_list<Scalar>& value) : vector_(vector), value_(value) {};

			~KernelVectorSetValue2() = default;

			bool operator()()
			{
				vector_.SetValue(value_);

				return true;
			}

		private:
			T& vector_;
			const std::initializer_list<Scalar>& value_;
		};

		template<typename T>
		class VectorKernel : public IKernel
		{
		public:
			using value_type = T;

			VectorKernel(const Device& device)
			{
				Initialize<T>(device);
			}

			template<typename U = T> 
			std::enable_if_t<std::is_same_v<U, eilig::Vector>, void> Initialize(const Device& device)
			{
				vector_ = T();
			}
			
			template<typename U = T> 
			std::enable_if_t<std::is_same_v<U, eilig::opencl::Vector>, void> Initialize(const Device& device)
			{
				vector_ = T(device.kernels);
			}

			~VectorKernel() = default;

			T& GetVector()
			{
				return vector_;
			}

			KernelVectorResize<T> GetKernelResize(NumberRows numberRows)
			{
				return KernelVectorResize<T>(vector_, numberRows);
			}
			KernelVectorCopyScalar<T> GetKernelCopyScalar(Scalar value)
			{
				return KernelVectorCopyScalar<T>(vector_, value);
			}
			KernelVectorCopyVector<T> GetKernelCopyVector(const eilig::Vector& value)
			{
				return KernelVectorCopyVector<T>(vector_, value);
			}
			KernelVectorCopyVector2<T> GetKernelCopyVector2(const T& value)
			{
				return KernelVectorCopyVector2<T>(vector_, value);
			}
			KernelVectorAddScalar<T> GetKernelAddScalar(Scalar value)
			{
				return KernelVectorAddScalar<T>(vector_, value);
			}
			KernelVectorAddVector<T> GetKernelAddVector(const T& value)
			{
				return KernelVectorAddVector<T>(vector_, value);
			}
			KernelVectorSubScalar<T> GetKernelSubScalar(Scalar value)
			{
				return KernelVectorSubScalar<T>(vector_, value);
			}
			KernelVectorSubVector<T> GetKernelSubVector(const T& value)
			{
				return KernelVectorSubVector<T>(vector_, value);
			}
			KernelVectorMulScalar<T> GetKernelMulScalar(Scalar value)
			{
				return KernelVectorMulScalar<T>(vector_, value);
			}
			KernelVectorSwapRows<T> GetKernelSwapRows(Index row1, Index row2)
			{
				return KernelVectorSwapRows<T>(vector_, row1, row2);
			}
			KernelVectorRegion<T> GetKernelRegion(const T& value, Index row1, Index row2)
			{
				return KernelVectorRegion<T>(vector_, value, row1, row2);
			}
			KernelVectorReplace1<T> GetKernelReplace1(const T& value, Index row1)
			{
				return KernelVectorReplace1<T>(vector_, value, row1);
			}
			KernelVectorReplace2<T> GetKernelReplace2(const eilig::Vector& value, Index row1)
			{
				return KernelVectorReplace2<T>(vector_, value, row1);
			}
			KernelVectorSetValue<T> GetKernelSetValue(const eilig::Vector& value)
			{
				return KernelVectorSetValue<T>(vector_, value);
			}
			KernelVectorSetValue2<T> GetKernelSetValue2(const std::initializer_list<Scalar>& value)
			{
				return KernelVectorSetValue2<T>(vector_, value);
			}
			KernelVectorInitializerList<T> GetKernelInitializerList(const std::initializer_list<Scalar>& value)
			{
				return KernelVectorInitializerList<T>(vector_, value);
			}

		private:
			T vector_;
		};

		//---------------------------------------------------------------------------

		template<typename T>
		class KernelEllpackResize
		{
		public:
			KernelEllpackResize(T& matrix, NumberRows numberRows, NumberCols numberCols) : matrix_(matrix), numberRows_(numberRows), numberCols_(numberCols) {};

			~KernelEllpackResize() = default;

			bool operator()()
			{
				matrix_.Resize(numberRows_, numberCols_);

				return true;
			}

		private:
			T& matrix_;

			NumberRows numberRows_;
			NumberCols numberCols_;
		};

		template<typename T>
		class KernelEllpackInitializerList
		{
		public:
			KernelEllpackInitializerList(T& matrix, const std::initializer_list<std::initializer_list<Scalar>>& value, const Block& block) : matrix_(matrix), value_(value), block_(block) {};

			~KernelEllpackInitializerList() = default;

			bool operator()()
			{
				matrix_.Equal(value_, block_.row, block_.numberRows);

				return true;
			}

		private:
			T& matrix_;

			const Block& block_;
			const std::initializer_list<std::initializer_list<Scalar>>& value_;
		};

		template<typename T>
		class KernelEllpackCopyScalar
		{
		public:
			KernelEllpackCopyScalar(T& matrix, Scalar value) : matrix_(matrix), value_(value) {};

			~KernelEllpackCopyScalar() = default;

			bool operator()()
			{
				matrix_.Equal(value_);

				return true;
			}

		private:
			T& matrix_;

			Scalar value_;
		};

		template<typename T>
		class KernelEllpackCopyMatrix
		{
		public:
			KernelEllpackCopyMatrix(T& matrix, const eilig::Ellpack& value, const Block& block) : matrix_(matrix), value_(value), block_(block) {};

			~KernelEllpackCopyMatrix() = default;

			bool operator()()
			{
				matrix_.Equal(value_, block_.row, block_.numberRows);

				return true;
			}

		private:
			T& matrix_;

			const eilig::Ellpack& value_;
			const Block& block_;
		};

		template<typename T>
		class KernelEllpackCopyMatrix2
		{
		public:
			KernelEllpackCopyMatrix2(T& matrix, const T& value) : matrix_(matrix), value_(value) {};

			~KernelEllpackCopyMatrix2() = default;

			bool operator()()
			{
				matrix_.Equal(value_);

				return true;
			}

		private:
			T& matrix_;

			const T& value_;
		};

		template<typename T>
		class KernelEllpackAddScalar
		{
		public:
			KernelEllpackAddScalar(T& matrix, Scalar value) : matrix_(matrix), value_(value) {};

			~KernelEllpackAddScalar() = default;

			bool operator()()
			{
				matrix_.Add(value_);

				return true;
			}

		private:
			T& matrix_;

			Scalar value_;
		};

		template<typename T>
		class KernelEllpackAddMatrix
		{
		public:
			KernelEllpackAddMatrix(T& matrix, const T& value) : matrix_(matrix), value_(value) {};

			~KernelEllpackAddMatrix() = default;

			bool operator()()
			{
				matrix_.Add(value_);

				return true;
			}

		private:
			T& matrix_;

			const T& value_;
		};

		template<typename T>
		class KernelEllpackSubScalar
		{
		public:
			KernelEllpackSubScalar(T& matrix, Scalar value) : matrix_(matrix), value_(value) {};

			~KernelEllpackSubScalar() = default;

			bool operator()()
			{
				matrix_.Sub(value_);

				return true;
			}

		private:
			T& matrix_;

			Scalar value_;
		};

		template<typename T>
		class KernelEllpackSubMatrix
		{
		public:
			KernelEllpackSubMatrix(T& matrix, const T& value) : matrix_(matrix), value_(value) {};

			~KernelEllpackSubMatrix() = default;

			bool operator()()
			{
				matrix_.Sub(value_);
				return true;
			}

		private:
			T& matrix_;

			const T& value_;
		};

		template<typename T>
		class KernelEllpackMulScalar
		{
		public:
			KernelEllpackMulScalar(T& matrix, Scalar value) : matrix_(matrix), value_(value) {};

			~KernelEllpackMulScalar() = default;

			bool operator()()
			{
				matrix_.Mul(value_);
				return true;
			}

		private:
			T& matrix_;

			Scalar value_;
		};

		template<typename T>
		class KernelEllpackMulVector
		{
		public:
			KernelEllpackMulVector(const T& matrix, const typename T::vector_type& value, const Block& block) : matrix_(matrix), value_(value), block_(block){};

			~KernelEllpackMulVector() = default;

			TaskBlock<typename T::vector_type> operator()()
			{
				return TaskBlock<typename T::vector_type>{matrix_ * value_, block_};
			}

		private:
			const T& matrix_;
			const Block& block_;
			const typename T::vector_type& value_;
		};

		template<typename T>
		class KernelEllpackSwapCols
		{
		public:
			KernelEllpackSwapCols(T& matrix, Index col1, Index col2) : matrix_(matrix), col1_(col1), col2_(col2) {};

			~KernelEllpackSwapCols() = default;

			bool operator()()
			{
				matrix_.SwapCols(col1_, col2_);
				return true;
			}

		private:
			T& matrix_;

			Index col1_;
			Index col2_;
		};

		template<typename T>
		class KernelEllpackTrace
		{
		public:
			KernelEllpackTrace(const T& matrix, const Block& block) : matrix_(matrix), block_(block) {};

			~KernelEllpackTrace() = default;

			Scalar operator()()
			{
				return matrix_.Trace(block_.row);
			}

		private:
			const T& matrix_;

			const Block& block_;
		};

		template<typename T>
		class KernelEllpackSum
		{
		public:
			KernelEllpackSum(const T& matrix) : matrix_(matrix){};

			~KernelEllpackSum() = default;

			Scalar operator()()
			{
				return matrix_.Sum();
			}

		private:
			const T& matrix_;
		};

		template<typename T>
		class KernelEllpackDiagonal
		{
		public:
			KernelEllpackDiagonal(const T& matrix, const Block& block) : matrix_(matrix), block_(block) {};

			~KernelEllpackDiagonal() = default;

			TaskBlock<T> operator()()
			{
				return TaskBlock<T>{matrix_.Diagonal(block_.row), block_};
			}

		private:
			const T& matrix_;

			const Block& block_;
		};

		template<typename T>
		class KernelEllpackDiagonalScale
		{
		public:
			KernelEllpackDiagonalScale(const T& matrix, const Scalar& factor, const Block& block) : matrix_(matrix), factor_(factor), block_(block) {};

			~KernelEllpackDiagonalScale() = default;

			TaskBlock<T> operator()()
			{
				return TaskBlock<T>{matrix_.DiagonalScale(factor_, block_.row), block_};
			}

		private:
			const T& matrix_;
			const Scalar& factor_;
			const Block& block_;
		};

		template<typename T>
		class KernelEllpackDiagonalVector
		{
		public:
			KernelEllpackDiagonalVector(const T& matrix, const Block& block) : matrix_(matrix), block_(block) {};

			~KernelEllpackDiagonalVector() = default;

			TaskBlock<typename T::vector_type> operator()()
			{
				return TaskBlock<typename T::vector_type>{matrix_.DiagonalVector(block_.row), block_};
			}

		private:
			const T& matrix_;
			const Block& block_;
		};

		template<typename T>
		class KernelEllpackLowerWithDiagonal
		{
		public:
			KernelEllpackLowerWithDiagonal(const T& matrix, const Block& block) : matrix_(matrix), block_(block) {};

			~KernelEllpackLowerWithDiagonal() = default;

			TaskBlock<T> operator()()
			{
				return TaskBlock<T>{matrix_.LowerWithDiagonal(block_.row), block_};
			}

		private:
			const T& matrix_;

			const Block& block_;
		};

		template<typename T>
		class KernelEllpackLowerWithoutDiagonal
		{
		public:
			KernelEllpackLowerWithoutDiagonal(const T& matrix, const Block& block) : matrix_(matrix), block_(block) {};

			~KernelEllpackLowerWithoutDiagonal() = default;

			TaskBlock<T> operator()()
			{
				return TaskBlock<T>{matrix_.LowerWithoutDiagonal(block_.row), block_};
			}

		private:
			const T& matrix_;

			const Block& block_;
		};

		template<typename T>
		class KernelEllpackUpperWithDiagonal
		{
		public:
			KernelEllpackUpperWithDiagonal(const T& matrix, const Block& block) : matrix_(matrix), block_(block) {};

			~KernelEllpackUpperWithDiagonal() = default;

			TaskBlock<T> operator()()
			{
				return TaskBlock<T>{matrix_.UpperWithDiagonal(block_.row), block_};
			}

		private:
			const T& matrix_;

			const Block& block_;
		};

		template<typename T>
		class KernelEllpackUpperWithoutDiagonal
		{
		public:
			KernelEllpackUpperWithoutDiagonal(const T& matrix, const Block& block) : matrix_(matrix), block_(block) {};

			~KernelEllpackUpperWithoutDiagonal() = default;

			TaskBlock<T> operator()()
			{
				return TaskBlock<T>{matrix_.UpperWithoutDiagonal(block_.row), block_};
			}

		private:
			const T& matrix_;

			const Block& block_;
		};

		template<typename T>
		class KernelEllpackReplace
		{
		public:
			KernelEllpackReplace(T& matrix, const eilig::Ellpack& value, Index row1, Index col1): matrix_(matrix), value_(value), row1_(row1), col1_(col1) {};

			~KernelEllpackReplace() = default;

			bool operator()()
			{
				matrix_.Replace(row1_, col1_, value_);

				return true;
			}

		private:
			T& matrix_;
			const eilig::Ellpack& value_;

			Index row1_;
			Index col1_;
		};

		template<typename T>
		class EllpackKernel : public IKernel
		{
		public:
			using value_type = T;

			EllpackKernel(const Device& device)
			{
				Initialize<T>(device);
			}

			template<typename U = T> 
			std::enable_if_t<std::is_same_v<U, eilig::Ellpack>, void> Initialize(const Device& device)
			{
				matrix_ = T();
			}
			
			template<typename U = T>
			std::enable_if_t<std::is_same_v<U, eilig::opencl::Ellpack>, void> Initialize(const Device& device)
			{
				matrix_ = T(device.kernels);
			}

			~EllpackKernel() = default;

			T& GetMatrix()
			{
				return matrix_;
			}

			KernelEllpackResize<T> GetKernelResize(NumberRows numberRows, NumberCols numberCols)
			{
				return KernelEllpackResize<T>(matrix_, numberRows, numberCols);
			}
			KernelEllpackCopyScalar<T> GetKernelCopyScalar(Scalar value)
			{
				return KernelEllpackCopyScalar<T>(matrix_, value);
			}
			KernelEllpackCopyMatrix<T> GetKernelCopyMatrix(const eilig::Ellpack& value, const Block& block)
			{
				return KernelEllpackCopyMatrix<T>(matrix_, value, block);
			}
			KernelEllpackCopyMatrix2<T> GetKernelCopyMatrix2(const T& value)
			{
				return KernelEllpackCopyMatrix2<T>(matrix_, value);
			}
			KernelEllpackAddScalar<T> GetKernelAddScalar(Scalar value)
			{
				return KernelEllpackAddScalar<T>(matrix_, value);
			}
			KernelEllpackAddMatrix<T> GetKernelAddMatrix(const T& value)
			{
				return KernelEllpackAddMatrix<T>(matrix_, value);
			}
			KernelEllpackSubScalar<T> GetKernelSubScalar(Scalar value)
			{
				return KernelEllpackSubScalar<T>(matrix_, value);
			}
			KernelEllpackSubMatrix<T> GetKernelSubMatrix(const T& value)
			{
				return KernelEllpackSubMatrix<T>(matrix_, value);
			}
			KernelEllpackMulScalar<T> GetKernelMulScalar(Scalar value)
			{
				return KernelEllpackMulScalar<T>(matrix_, value);
			}
			KernelEllpackMulVector<T> GetKernelMulVector(const typename T::vector_type& value, const Block& block)
			{
				return KernelEllpackMulVector<T>(matrix_, value, block);
			}
			KernelEllpackSwapCols<T> GetKernelSwapCols(Index col1, Index col2)
			{
				return KernelEllpackSwapCols<T>(matrix_, col1, col2);
			}
			KernelEllpackTrace<T> GetKernelTrace(const Block& block)
			{
				return KernelEllpackTrace<T>(matrix_, block);
			}
			KernelEllpackSum<T> GetKernelSum()
			{
				return KernelEllpackSum<T>(matrix_);
			}
			KernelEllpackDiagonal<T> GetKernelDiagonal(const Block& block)
			{
				return KernelEllpackDiagonal<T>(matrix_, block);
			}
			KernelEllpackDiagonalScale<T> GetKernelDiagonalScale(const Scalar& factor, const Block& block)
			{
				return KernelEllpackDiagonalScale<T>(matrix_, factor, block);
			}
			KernelEllpackDiagonalVector<T> GetKernelDiagonalVector(const Block& block)
			{
				return KernelEllpackDiagonalVector<T>(matrix_, block);
			}
			KernelEllpackLowerWithDiagonal<T> GetKernelLowerWithDiagonal(const Block& block)
			{
				return KernelEllpackLowerWithDiagonal<T>(matrix_, block);
			}
			KernelEllpackLowerWithoutDiagonal<T> GetKernelLowerWithoutDiagonal(const Block& block)
			{
				return KernelEllpackLowerWithoutDiagonal<T>(matrix_, block);
			}
			KernelEllpackUpperWithDiagonal<T> GetKernelUpperWithDiagonal(const Block& block)
			{
				return KernelEllpackUpperWithDiagonal<T>(matrix_, block);
			}
			KernelEllpackUpperWithoutDiagonal<T> GetKernelUpperWithoutDiagonal(const Block& block)
			{
				return KernelEllpackUpperWithoutDiagonal<T>(matrix_, block);
			}
			KernelEllpackReplace<T> GetKernelReplace(const eilig::Ellpack& value, Index row1, Index col1)
			{
				return KernelEllpackReplace<T>(matrix_, value, row1, col1);
			}
			KernelEllpackInitializerList<T> GetKernelInitializerList(const std::initializer_list<std::initializer_list<Scalar>>& value, const Block& block)
			{
				return KernelEllpackInitializerList<T>(matrix_, value, block);
			}

		private:
			T matrix_;
		};
    }
} /* namespace eilig */

#endif /* EILIG_THREADED_HPP_ */

/*
void WaitForAll_Ellpack_Matrix_Multiplication(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL, Ellpack& result)
{
	while (!queue.empty())
	{
		const auto& it = queue.begin();

		while (it != queue.end())
		{
			if (IsReady(*it))
			{
				auto taskBlock = it->get();
				const auto& value = taskBlock.value;

				std::cout << eilig::ListMatrix(value) << std::endl;

				queue.erase(it);
				break;
			}
		}
	}

	while (!queueCL.empty())
	{
		const auto& it = queueCL.begin();

		while (it != queueCL.end())
		{
			if (IsReady(*it))
			{
				auto taskBlock = it->get();

				queueCL.erase(it);
				break;
			}
		}
	}
}

template<typename T>
class KernelEllpackMulMatrix
{
public:
	KernelEllpackMulMatrix(T& matrix, const T& lhs, const eilig::Ellpack& rhs,const Block& block1, const Block& block2) : matrix_(matrix), lhs_(lhs), rhs_(rhs), block1_(block1), block2_(block2) {};

	~KernelEllpackMulMatrix() = default;

	TaskBlock<T> operator()()
	{
		return TaskBlock<T>{lhs_.Mul(rhs_, block2_.row, block2_.numberRows), block1_};
	}

private:
	T& matrix_;

	const Block& block1_;
	const Block& block2_;

	const T& lhs_;
	const eilig::Ellpack& rhs_;
};

KernelEllpackMulMatrix<T> GetKernelMulMatrix(const T& lhs, const eilig::Ellpack& rhs,const Block& block1, const Block& block2)
{
	return KernelEllpackMulMatrix<T>(matrix_, lhs, rhs, block1, block2);
}
*/