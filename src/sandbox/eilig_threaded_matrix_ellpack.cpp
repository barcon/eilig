#include "eilig_threaded_matrix_ellpack.hpp"
#include "eilig_routines.hpp"

namespace eilig
{
    namespace threaded
    {
        extern ThreadPool threadPool;

        Ellpack::Ellpack(const Ellpack& input)
        {
            (*this) = input;
        }
        Ellpack::Ellpack(const Devices& devices)
        { 
            SetDevices(devices);
            Resize(1, 1);
        }
        Ellpack::Ellpack(const Devices& devices, const std::initializer_list<std::initializer_list<Scalar>>& value)
        {
            TaskQueueBool queue;

            SetDevices(devices);
            Resize(value.size(), value.begin()->size());

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelInitializerList(value, blocks_[i]);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelInitializerList(value, blocks_[i]);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

			WaitForAll(queue);
        }
        Ellpack::Ellpack(const Devices& devices, const eilig::Ellpack& input)
        {
            TaskQueueBool queue;

            SetDevices(devices);
            Resize(input.GetRows(), input.GetCols());

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if(!blocks_[i].isUsed)
                {
                    continue;
				}

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyMatrix(input, blocks_[i]);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyMatrix(input, blocks_[i]);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        Ellpack::Ellpack(const Devices& devices, const eilig::Matrix& input)
        {
            SetDevices(devices);
            Resize(input.GetRows(), input.GetCols());

            for (Index i = 0; i < numberRows_; i++)
            {
                for (Index j = 0; j < numberCols_; j++)
                {
                    if (utils::math::IsAlmostEqual(input(i, j), 0.0, 5))
                    {
                        continue;
                    }

                    (*this)(i, j) = input(i, j);
                }
            }
        }
        Ellpack::Ellpack(const Devices& devices, NumberRows numberRows, NumberCols numberCols)
        {
            SetDevices(devices);
            Resize(numberRows, numberCols);
        }
        Ellpack::Ellpack(const Devices& devices, NumberRows numberRows, NumberCols numberCols, Type type)
        {
            SetDevices(devices);
            switch (type)
            {
            case matrix_ones:
                Resize(numberRows, numberCols, 1.0);
                break;
            case matrix_diagonal:
                Resize(numberRows, numberCols, 0.0);

                for (Index i = 0; (i < numberRows) && (i < numberCols); ++i)
                {
                    (*this)(i, i) = 1.0;
                }
                break;
            case matrix_zeros:
            default:
                Resize(numberRows, numberCols, 0.0);
            }
        }
        Ellpack::Ellpack(Ellpack&& input) noexcept
        {
            (*this) = std::move(input);
        }

        eilig::Ellpack Ellpack::Convert() const
        {
			eilig::Ellpack res(numberRows_, numberCols_);

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    res.Replace(blocks_[i].row, 0, kernel->GetMatrix());
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    res.Replace(blocks_[i].row, 0, kernel->GetMatrix().Convert());
                }
            }

            return res;
        }

        void Ellpack::Resize(NumberRows numberRows, NumberCols numberCols)
        {
            TaskQueueBool queue;

            NumberRows blockSize{ 0 };
			NumberRows restSize{ 0 };

            if (numberRows == 0 || numberCols == 0)
            {
                throw std::invalid_argument("Matrix dimensions cannot be zero.");
            }

            if (numberRows == numberRows_ && numberCols == numberCols_)
            {
                return;
            }

			numberRows_ = numberRows;
			numberCols_ = numberCols;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
			    AdjustBlock(blocks_[i], numberRows_, blocks_.size(), i);

                if(!blocks_[i].isUsed)
                {
                    continue;
				}

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelResize(blocks_[i].numberRows, numberCols_);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelResize(blocks_[i].numberRows, numberCols_);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Ellpack::Resize(NumberRows numberRows, NumberRows numberCols, Scalar value)
        {
            Resize(numberRows, numberCols);
            (*this) = value;
        }
        void Ellpack::Fill(Scalar value)
        {
            (*this) = value;
        }

        EntryProxyEllpack Ellpack::operator()(Index row, Index col)
        {
            auto [blockIndex, rowIndex] = GetOffset(blocks_, row);

           return EntryProxyEllpack(blocks_[blockIndex], rowIndex, col);
        }

        Ellpack& Ellpack::operator=(Scalar rhs)
        {
            Equal(rhs);

            return *this;
        }
        Ellpack& Ellpack::operator=(const Ellpack& rhs)
        {
            Equal(rhs);

            return *this;
        }
        Ellpack& Ellpack::operator=(Ellpack&& rhs) noexcept
        {
            if (&rhs == this)
            {
                return *this;
            }

            numberRows_ = rhs.numberRows_;
            numberCols_ = rhs.numberCols_;
            blocks_ = std::move(rhs.blocks_);

            return *this;
        }

        Ellpack Ellpack::operator+(Scalar rhs) const
        {
            TaskQueueBool queue;
            Ellpack res(*this);

            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return res;
        }
        Ellpack Ellpack::operator+(const Ellpack& rhs) const
        {
            TaskQueueBool queue;
            Ellpack res(*this);

            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(res.blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<EllpackKernelCPU>(rhs.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddMatrix(kernel2->GetMatrix());

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(res.blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<EllpackKernelGPU>(rhs.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddMatrix(kernel2->GetMatrix());

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return res;
        }
        Ellpack Ellpack::operator+() const
        {
            return (*this);
        }
        Ellpack operator+(Scalar lhs, const Ellpack& rhs)
        {
            return rhs + lhs;
        }

        Ellpack Ellpack::operator-(Scalar rhs) const
        {
            TaskQueueBool queue;
            Ellpack res(*this);

            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return res;
        }
        Ellpack Ellpack::operator-(const Ellpack& rhs) const
        {
            TaskQueueBool queue;
            Ellpack res(*this);

            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(res.blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<EllpackKernelCPU>(rhs.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubMatrix(kernel2->GetMatrix());

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(res.blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<EllpackKernelGPU>(rhs.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubMatrix(kernel2->GetMatrix());

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return res;
        }
        Ellpack Ellpack::operator-() const
        {
            return -1.0 * (*this);
        }
        Ellpack operator-(Scalar lhs, const Ellpack& rhs)
        {
            return -rhs + lhs;
        }

        Ellpack Ellpack::operator*(Scalar rhs) const
        {
            TaskQueueBool queue;
            Ellpack res(*this);

            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelMulScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelMulScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return res;
        }
        Vector Ellpack::operator*(const Vector& rhs) const
        {
            TaskQueueVector queue;
            TaskQueueVectorCL queueCL;

			Vector res(GetDevices(blocks_), numberRows_);

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel1 = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
					auto kernel2 = std::static_pointer_cast<VectorKernelCPU>(rhs.blocks_[i].kernel);
                    auto task = kernel1->GetKernelMulVector(kernel2->GetVector(), blocks_[i]);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel1 = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<VectorKernelGPU>(rhs.blocks_[i].kernel);
                    auto task = kernel1->GetKernelMulVector(kernel2->GetVector(), blocks_[i]);

                    queueCL.emplace_back(threadPool.submit_task(task));
                }
            }
            
            WaitForAll_Ellpack_Vector_Multiplication(queue, queueCL, res);
            
            return res;
        }
        Ellpack operator*(Scalar lhs, const Ellpack& rhs)
        {
            return rhs * lhs;
        }

        Ellpack& Ellpack::SwapRows(Index row1, Index row2)
        {
            TaskQueueBool queue;

            eilig::Ellpack region1;
            eilig::Ellpack region2;

			auto [blockIndex1, rowIndex1] = GetOffset(blocks_, row1);
			auto [blockIndex2, rowIndex2] = GetOffset(blocks_, row2);

            //------------------------------------------------------------------------------------------------------
            if (blocks_[blockIndex1].device.accelerator == Accelerator::cpu)
            {
                auto kernel1 = std::static_pointer_cast<EllpackKernelCPU>(blocks_[blockIndex1].kernel);				
                region1 = kernel1->GetMatrix().Region(rowIndex1, 0, rowIndex1, numberCols_ - 1);
            }
            else if (blocks_[blockIndex1].device.accelerator == Accelerator::gpu)
            {
                auto kernel1 = std::static_pointer_cast<EllpackKernelGPU>(blocks_[blockIndex1].kernel);
                region1 = kernel1->GetMatrix().Region(rowIndex1, 0, rowIndex1, numberCols_ - 1).Convert();
            }

            if (blocks_[blockIndex2].device.accelerator == Accelerator::cpu)
            {
                auto kernel2 = std::static_pointer_cast<EllpackKernelCPU>(blocks_[blockIndex2].kernel);
                region2 = kernel2->GetMatrix().Region(rowIndex2, 0, rowIndex2, numberCols_ - 1);
            }
            else if (blocks_[blockIndex2].device.accelerator == Accelerator::gpu)
            {
                auto kernel2 = std::static_pointer_cast<EllpackKernelGPU>(blocks_[blockIndex2].kernel);
                region2 = kernel2->GetMatrix().Region(rowIndex2, 0, rowIndex2, numberCols_ - 1).Convert();
            }
            //------------------------------------------------------------------------------------------------------

            if (blocks_[blockIndex1].device.accelerator == Accelerator::cpu)
            {
                auto kernel1 = std::static_pointer_cast<EllpackKernelCPU>(blocks_[blockIndex1].kernel);
                auto task = kernel1->GetKernelReplace(region2, rowIndex1, 0);

                queue.emplace_back(threadPool.submit_task(task));
            }
            else if (blocks_[blockIndex1].device.accelerator == Accelerator::gpu)
            {
                auto kernel1 = std::static_pointer_cast<EllpackKernelGPU>(blocks_[blockIndex1].kernel);
                auto task = kernel1->GetKernelReplace(region2, rowIndex1, 0);

                queue.emplace_back(threadPool.submit_task(task));
			}

            if (blocks_[blockIndex2].device.accelerator == Accelerator::cpu)
            {
                auto kernel2 = std::static_pointer_cast<EllpackKernelCPU>(blocks_[blockIndex2].kernel);
                auto task = kernel2->GetKernelReplace(region1, rowIndex2, 0);
             
                queue.emplace_back(threadPool.submit_task(task));
            }
            else if (blocks_[blockIndex2].device.accelerator == Accelerator::gpu)
            {
                auto kernel2 = std::static_pointer_cast<EllpackKernelGPU>(blocks_[blockIndex2].kernel);
                auto task = kernel2->GetKernelReplace(region1, rowIndex2, 0);
                
                queue.emplace_back(threadPool.submit_task(task));
			}
            
            WaitForAll(queue);

            return *this;
        }
        Ellpack& Ellpack::SwapCols(Index col1, Index col2)
        {
            TaskQueueBool queue;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelSwapCols(col1, col2);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelSwapCols(col1, col2);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return *this;
        }
        Scalar Ellpack::Trace() const
        {
            TaskQueueScalar queue;
			Scalar res{ 0.0 };

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelTrace(blocks_[i]);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelTrace(blocks_[i]);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            return WaitForAll_Ellpack_Sum(queue);
        }
        Scalar Ellpack::Sum() const
        {
            TaskQueueScalar queue;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelSum();

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelSum();

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            return WaitForAll_Ellpack_Sum(queue);
        }
        Ellpack Ellpack::Diagonal() const
        {
            TaskQueueEllpack queue;
            TaskQueueEllpackCL queueCL;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelDiagonal(blocks_[i]);
                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelDiagonal(blocks_[i]);
                    queueCL.emplace_back(threadPool.submit_task(task));
                }
            }
			return WaitForAll_Ellpack_Diagonal(queue, queueCL);
        }
        Ellpack Ellpack::DiagonalScale(Scalar factor) const
        {
            TaskQueueEllpack queue;
            TaskQueueEllpackCL queueCL;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelDiagonalScale(factor, blocks_[i]);
                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelDiagonalScale(factor, blocks_[i]);
                    queueCL.emplace_back(threadPool.submit_task(task));
                }
            }
            return WaitForAll_Ellpack_Diagonal_Scale(queue, queueCL);
        }
        Vector  Ellpack::DiagonalVector() const
        {
            TaskQueueVector queue;
            TaskQueueVectorCL queueCL;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelDiagonalVector(blocks_[i]);
                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelDiagonalVector(blocks_[i]);
                    queueCL.emplace_back(threadPool.submit_task(task));
                }
            }
            return WaitForAll_Ellpack_Diagonal_Vector(queue, queueCL);
        }
        Ellpack Ellpack::Lower(bool diag) const
        {
            if (diag)
            {
                return LowerWithDiagonal();
            }
            else
            {
                return LowerWithoutDiagonal();
            }
        }
        Ellpack Ellpack::LowerWithDiagonal() const
        {
            TaskQueueEllpack queue;
            TaskQueueEllpackCL queueCL;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelLowerWithDiagonal(blocks_[i]);
                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelLowerWithDiagonal(blocks_[i]);
                    queueCL.emplace_back(threadPool.submit_task(task));
                }
            }
            return WaitForAll_Ellpack_Lower_With_Diagonal(queue, queueCL);
        }
        Ellpack Ellpack::LowerWithoutDiagonal() const
        {
            TaskQueueEllpack queue;
            TaskQueueEllpackCL queueCL;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelLowerWithoutDiagonal(blocks_[i]);
                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelLowerWithoutDiagonal(blocks_[i]);
                    queueCL.emplace_back(threadPool.submit_task(task));
                }
            }
            return WaitForAll_Ellpack_Lower_Without_Diagonal(queue, queueCL);
        }
        Ellpack Ellpack::Upper(bool diag) const
        {
            if (diag)
            {
                return UpperWithDiagonal();
            }
            else
            {
                return UpperWithoutDiagonal();
            }
        }
        Ellpack Ellpack::UpperWithDiagonal() const
        {
            TaskQueueEllpack queue;
            TaskQueueEllpackCL queueCL;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelUpperWithDiagonal(blocks_[i]);
                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelUpperWithDiagonal(blocks_[i]);
                    queueCL.emplace_back(threadPool.submit_task(task));
                }
            }
            return WaitForAll_Ellpack_Upper_With_Diagonal(queue, queueCL);
        }
        Ellpack Ellpack::UpperWithoutDiagonal() const
        {
            TaskQueueEllpack queue;
            TaskQueueEllpackCL queueCL;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelUpperWithoutDiagonal(blocks_[i]);
                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelUpperWithoutDiagonal(blocks_[i]);
                    queueCL.emplace_back(threadPool.submit_task(task));
                }
            }
            return WaitForAll_Ellpack_Upper_Without_Diagonal(queue, queueCL);
        }
        Ellpack Ellpack::Region(Index row1, Index col1, Index row2, Index col2) const
        {
            TaskQueueBool queue;
            Ellpacks ellpacks;

            Index numberRows = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
            Index numberCols = col1 <= col2 ? (col2 - col1) + 1 : (col1 - col2) + 1;

            Index rowStart{ 0 };
            Index rowEnd{ 0 };
           
            Index colStart{ 0 };
            Index colEnd{ 0 };

            if ((row1 <= row2) && (col1 <= col2))
            {
                rowStart = row1;
				rowEnd = row2;

                colStart = col1;
				colEnd = col2;
            }
            else if ((row1 >= row2) && (col1 <= col2))
            {
                rowStart = row2;
				rowStart = row1;

                colStart = col1;
				colEnd = col2;
            }
            else if ((row1 >= row2) && (col1 >= col2))
            {
                rowStart = row2;
				rowEnd = row1;

                colStart = col2;
				colEnd = col1;
            }
            else
            {
                rowStart = row1;
				rowEnd = row2;

                colStart = col2;
				colEnd = col1;
            }

            while(true)
            {
                eilig::Ellpack ellpack;

                auto [i, rowIndex1] = GetOffset(blocks_, rowStart);

                auto delta1 = blocks_[i].numberRows - rowIndex1;
				auto delta2 = rowEnd - rowStart + 1;
                auto delta = std::min(delta1, delta2);

                if (delta == 0)
                {
                    break;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    ellpack = kernel->GetMatrix().Region(rowIndex1, colStart, rowIndex1 + delta - 1, colEnd);
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    ellpack = kernel->GetMatrix().Convert().Region(rowIndex1, colStart, rowIndex1 + delta - 1, colEnd);
                }

                ellpacks.emplace_back(ellpack);

                rowStart += delta;
            }

            return Ellpack(GetDevices(blocks_), eilig::Ellpack(ellpacks));
        }

        NumberRows Ellpack::GetRows() const
        {
            return numberRows_;
        }
        NumberCols Ellpack::GetCols() const
        {
            return numberCols_;
        }
        Scalar Ellpack::GetValue(Index row, Index col) const
        {
            Scalar res{ 0.0 };

			auto [blockIndex, rowIndex] = GetOffset(blocks_, row);

            if (blocks_[blockIndex].device.accelerator == Accelerator::cpu)
            {
                res = std::static_pointer_cast<EllpackKernelCPU>(blocks_[blockIndex].kernel)->GetMatrix().GetValue(rowIndex, col);
            }
            else if (blocks_[blockIndex].device.accelerator == Accelerator::gpu)
            {
                res = std::static_pointer_cast<EllpackKernelGPU>(blocks_[blockIndex].kernel)->GetMatrix().GetValue(rowIndex, col);
            }

            return res;
        }

        void Ellpack::Equal(Index row, Index col, Scalar value)
        {
            (*this)(row, col) = value;
        }
        void Ellpack::Equal(Scalar value)
        {       
            TaskQueueBool queue;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Ellpack::Equal(const Ellpack& value)
        {
            TaskQueueBool queue;

            SetDevices(GetDevices(value.blocks_));
            Resize(value.numberRows_, value.numberCols_);

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<EllpackKernelCPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyMatrix2(kernel2->GetMatrix());

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<EllpackKernelGPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyMatrix2(kernel2->GetMatrix());

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Ellpack::Equal(const std::initializer_list<std::initializer_list<Scalar>>& value)
        {
            TaskQueueBool queue;

            Resize(value.size(), value.begin()->size());

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelInitializerList(value, blocks_[i]);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelInitializerList(value, blocks_[i]);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Ellpack::Add(Scalar value)
        {
            TaskQueueBool queue;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelAddScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelAddScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Ellpack::Add(const Ellpack& value)
        {
            TaskQueueBool queue;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<EllpackKernelCPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddMatrix(kernel2->GetMatrix());

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<EllpackKernelGPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddMatrix(kernel2->GetMatrix());

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Ellpack::Sub(Scalar value)
        {
            TaskQueueBool queue;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelSubScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelSubScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Ellpack::Sub(const Ellpack& value)
        {
            TaskQueueBool queue;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<EllpackKernelCPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubMatrix(kernel2->GetMatrix());

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<EllpackKernelGPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubMatrix(kernel2->GetMatrix());

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Ellpack::Mul(Scalar value)
        {
            TaskQueueBool queue;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelMulScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelMulScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }

        const Blocks& Ellpack::GetBlocks() const
        {
            return blocks_;
        }

        void Ellpack::SetDevices(const Devices& devices)
        {
            if (devices.size() == 0)
            {
                throw std::invalid_argument("Number of devices cannot be zero.");
            }

            blocks_.clear();

            for (Index i = 0; i < devices.size(); ++i)
            {
                IKernelPtr kernel{ nullptr };

                if (devices[i].accelerator == Accelerator::cpu)
                {
                    kernel = std::make_shared<EllpackKernelCPU>(devices[i]);
                }
                else if (devices[i].accelerator == Accelerator::gpu)
                {
                    kernel = std::make_shared<EllpackKernelGPU>(devices[i]);
                }

                auto block = Block();
                
                block.isUsed = false;
				block.index = i;
				block.device = devices[i];
				block.kernel = kernel;
				block.row = 0;
				block.numberRows = 0;

                blocks_.emplace_back(block);
            }
        }
    }
} /* namespace eilig */


        /*Ellpack Ellpack::operator*(const Ellpack& rhs) const
        {
            TaskQueueEllpack queue;
            TaskQueueEllpackCL queueCL;

            Ellpack res(GetDevices(blocks_), numberRows_, rhs.numberCols_);

            for (Index i = 0; i < rhs.blocks_.size(); ++i)
            {
                if (!rhs.blocks_[i].isUsed)
                {
                    continue;
                }

                eilig::Ellpack matrix;

                if (rhs.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelCPU>(rhs.blocks_[i].kernel);

                    matrix = kernel->GetMatrix();
                }
                else if (rhs.blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<EllpackKernelGPU>(rhs.blocks_[i].kernel);

                    matrix = kernel->GetMatrix().Convert();
                }

                for (Index j = 0; j < res.blocks_.size(); ++j)
                {
                    if (!res.blocks_[j].isUsed)
                    {
                        continue;
                    }

                    if (res.blocks_[j].device.accelerator == Accelerator::cpu)
                    {
                        auto kernel = std::static_pointer_cast<EllpackKernelCPU>(res.blocks_[j].kernel);
                        auto kernel2 = std::static_pointer_cast<EllpackKernelCPU>(blocks_[j].kernel);
                        auto task = kernel->GetKernelMulMatrix(kernel2->GetMatrix(), matrix, res.blocks_[j], rhs.blocks_[i]);

                        queue.emplace_back(threadPool.submit_task(task));

                    }
                    else if (res.blocks_[j].device.accelerator == Accelerator::gpu)
                    {
                        auto kernel = std::static_pointer_cast<EllpackKernelGPU>(res.blocks_[j].kernel);
                        auto kernel2 = std::static_pointer_cast<EllpackKernelGPU>(blocks_[j].kernel);
                        auto task = kernel->GetKernelMulMatrix(kernel2->GetMatrix(), matrix, res.blocks_[j], rhs.blocks_[i]);

                        queueCL.emplace_back(threadPool.submit_task(task));
                    }
                }

                WaitForAll_Ellpack_Matrix_Multiplication(queue, queueCL, res);
            }

            return res;
        }*/