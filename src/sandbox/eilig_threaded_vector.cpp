#include "eilig_threaded_vector.hpp"

#include "eilig_routines.hpp"

namespace eilig
{
    namespace threaded
    {
        extern ThreadPool threadPool;

        Vector::Vector(const Vector& input)
        {
            (*this) = input;
        }
        Vector::Vector(const Devices& devices)
        {
            SetDevices(devices);
            Resize(1);
        }
        Vector::Vector(const Devices& devices, const std::initializer_list<Scalar>& value)
        {
            TaskQueueBool queue;

            SetDevices(devices);
            Resize(value.size());
           
            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelInitializerList(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelInitializerList(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

             WaitForAll(queue);
        }
        Vector::Vector(const Devices& devices, const eilig::Vector& input)
        {
            TaskQueueBool queue;

            SetDevices(devices);
            Resize(input.GetRows());

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyVector(input);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyVector(input);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

             WaitForAll(queue);
        }
        Vector::Vector(const Devices& devices, NumberRows numberRows)
        {
            SetDevices(devices);
            Resize(numberRows);
        }
        Vector::Vector(const Devices& devices, NumberRows numberRows, Scalar value)
        {
            SetDevices(devices);
            Resize(numberRows, value);
        }
        Vector::Vector(Vector&& input) noexcept
        {
            (*this) = std::move(input);
        }

        eilig::Vector Vector::Convert() const
        {
            eilig::Vector res(numberRows_);

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    res.Replace(blocks_[i].row, kernel->GetVector());
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    res.Replace(blocks_[i].row, kernel->GetVector().Convert());
                }
			}

            return res;
        }

        void Vector::Resize(NumberRows numberRows)
        {
            TaskQueueBool queue;

            if (numberRows == 0)
            {
                throw std::invalid_argument("Vector dimensions cannot be zero.");
            }

            if (numberRows_ == numberRows)
            {
                return;
            }

            numberRows_ = numberRows;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

				blocks_[i].numberRows = numberRows_;

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelResize(numberRows);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelResize(numberRows);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

             WaitForAll(queue);
        }
        void Vector::Resize(NumberRows numberRows, Scalar value)
        {
            Resize(numberRows);
            (*this) = value;
        }
        void Vector::Fill(Scalar value)
        {
            (*this) = value;
        }

        EntryProxyVector Vector::operator()(Index row)
        {
            return EntryProxyVector(blocks_, row);
        }

        Vector& Vector::operator=(Scalar rhs)
        {
            Equal(rhs);

            return *this;
        }
        Vector& Vector::operator=(const Vector& rhs)
        {
            Equal(rhs);

            return *this;
        }        
        Vector& Vector::operator=(Vector&& rhs) noexcept
        {
            if (&rhs == this)
            {
                return *this;
            }

            numberRows_ = rhs.numberRows_;
            blocks_ = std::move(rhs.blocks_);

            return *this;
        }

        Vector Vector::operator+(Scalar rhs) const
        {
            TaskQueueBool queue;

            Vector res(*this);

            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

			WaitForAll(queue);

            return res;
        }
        Vector Vector::operator+(const Vector& rhs) const
        {
            TaskQueueBool queue;

            Vector res(*this);

            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(res.blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<VectorKernelCPU>(rhs.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddVector(kernel2->GetVector());

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(res.blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<VectorKernelGPU>(rhs.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddVector(kernel2->GetVector());

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return res;
        }
        Vector Vector::operator+() const
        {
            return (*this);
        }
        Vector operator+(Scalar lhs, const Vector& rhs)
        {
            return rhs + lhs;
        }

        Vector Vector::operator-(Scalar rhs) const
        {
            TaskQueueBool queue;

            Vector res(*this);

            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return res;
        }
        Vector Vector::operator-(const Vector& rhs) const
        {
            TaskQueueBool queue;

            Vector res(*this);

            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(res.blocks_[i].kernel);
					auto kernel2 = std::static_pointer_cast<VectorKernelCPU>(rhs.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubVector(kernel2->GetVector());

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(res.blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<VectorKernelGPU>(rhs.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubVector(kernel2->GetVector());

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return res;
        }
        Vector Vector::operator-() const
        {
            return -1.0 * (*this);
        }
        Vector operator-(Scalar lhs, const Vector& rhs)
        {
            return -rhs + lhs;
        }

        Vector Vector::operator*(Scalar rhs) const
        {
            TaskQueueBool queue;

            Vector res(*this);

            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelMulScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(res.blocks_[i].kernel);
                    auto task = kernel->GetKernelMulScalar(rhs);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return res;
        }
        Vector operator*(Scalar lhs, const Vector& rhs)
        {
            return rhs * lhs;
        }

        Vector& Vector::SwapRows(Index row1, Index row2)
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
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelSwapRows(row1, row2);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelSwapRows(row1, row2);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

             WaitForAll(queue);

            return *this;
        }
        Vector  Vector::Region(Index row1, Index row2) const
        {          
            TaskQueueBool queue;

            Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
            Index aux2 = row1 <= row2 ? row1 : row2;
            Vector res(GetDevices(blocks_), aux1);
            
            for (Index i = 0; i < res.blocks_.size(); ++i)
            {
                if (!res.blocks_[i].isUsed)
                {
                    continue;
                }

                if (res.blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(res.blocks_[i].kernel);
                    auto& vector = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel)->GetVector();
                    auto task = kernel->GetKernelRegion(vector, row1, row2);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(res.blocks_[i].kernel);
                    auto& vector = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel)->GetVector();
                    auto task = kernel->GetKernelRegion(vector, row1, row2);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return res;
        }
        void Vector::Replace(Index row1, const Vector& in)
        {
            TaskQueueBool queue;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto& vector = std::static_pointer_cast<VectorKernelCPU>(in.blocks_[i].kernel)->GetVector();
                    auto task = kernel->GetKernelReplace1(vector, row1);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto& vector = std::static_pointer_cast<VectorKernelGPU>(in.blocks_[i].kernel)->GetVector();
                    auto task = kernel->GetKernelReplace1(vector, row1);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

             WaitForAll(queue);

            return;
        }
        void Vector::Replace(Index row1, const eilig::Vector& in)
        {
            TaskQueueBool queue;

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelReplace2(in, row1);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelReplace2(in, row1);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);

            return;
        }

        NumberRows Vector::GetRows() const
        {
            return numberRows_;
        }
        NumberCols Vector::GetCols() const
        {
            return 1;
        }
        Scalar Vector::GetValue(Index row) const
        {
			Scalar res{ 0.0 };

            if (blocks_[0].device.accelerator == Accelerator::cpu)
            {
                res = std::static_pointer_cast<VectorKernelCPU>(blocks_[0].kernel)->GetVector().GetValue(row);
            }
            else if (blocks_[0].device.accelerator == Accelerator::gpu)
            {
                res = std::static_pointer_cast<VectorKernelGPU>(blocks_[0].kernel)->GetVector().GetValue(row);
            }

            return res;
        }

        void Vector::Equal(Index i, Scalar value)
        {
            (*this)(i) = value;
        }
        void Vector::Equal(Scalar value)
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
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Vector::Equal(const Vector& value)
        {
            TaskQueueBool queue;

            SetDevices(GetDevices(value.blocks_));
            Resize(value.numberRows_);

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<VectorKernelCPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyVector2(kernel2->GetVector());

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<VectorKernelGPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyVector2(kernel2->GetVector());

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Vector::Equal(const eilig::Vector& value)
        {
            TaskQueueBool queue;

            Resize(value.GetRows());

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyVector(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelCopyVector(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Vector::Equal(const std::initializer_list<Scalar>& value)
        {
            TaskQueueBool queue;

            Resize(value.size());

            for (Index i = 0; i < blocks_.size(); ++i)
            {
                if (!blocks_[i].isUsed)
                {
                    continue;
                }

                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelInitializerList(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelInitializerList(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

             WaitForAll(queue);
        }

        void Vector::Add(Scalar value)
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
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelAddScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelAddScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Vector::Add(const Vector& value)
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
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<VectorKernelCPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddVector(kernel2->GetVector());

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<VectorKernelGPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelAddVector(kernel2->GetVector());

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Vector::Sub(Scalar value)
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
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelSubScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelSubScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Vector::Sub(const Vector& value)
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
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<VectorKernelCPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubVector(kernel2->GetVector());

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto kernel2 = std::static_pointer_cast<VectorKernelGPU>(value.blocks_[i].kernel);
                    auto task = kernel->GetKernelSubVector(kernel2->GetVector());

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }
        void Vector::Mul(Scalar value)
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
                    auto kernel = std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelMulScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    auto kernel = std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel);
                    auto task = kernel->GetKernelMulScalar(value);

                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            WaitForAll(queue);
        }

        const Blocks& Vector::GetBlocks() const
        {
            return blocks_;
        }

        void Vector::SetDevices(const Devices& devices)
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
                    kernel = std::make_shared<VectorKernelCPU>(devices[i]);
                }
                else if (devices[i].accelerator == Accelerator::gpu)
                {
                    kernel = std::make_shared<VectorKernelGPU>(devices[i]);
                }
				
                auto block = Block();

				block.isUsed = true;
                block.index = i;
                block.device = devices[i];
                block.kernel = kernel;
                block.row = 0;
                block.numberRows = numberRows_;

                blocks_.emplace_back(block);
            }
        }
    }
} /* namespace eilig */