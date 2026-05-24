#include "eilig_threaded_vector.hpp"
#include "eilig_threaded_vector_cpu.hpp"

extern ThreadPool threadPool;

namespace eilig
{
    namespace threaded
    {
        Vector::Vector(const Devices& devices)
        {
			SetDevices(devices);
            Resize(1);
        }
        Vector::Vector(const Devices& devices, const std::initializer_list<Scalar>& values)
        {
            SetDevices(devices);
            Resize(values.size());

            std::deque<std::future<bool>> queue;

            for (Index i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i]->GetType() == device_vector_cpu)
                {
                    auto task = std::static_pointer_cast<VectorCPU>(devices_[i])->GetKernelVectorInitializerList(values);
                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            while (!queue.empty())
            {
                const auto& it = queue.begin();

                while (it != queue.end())
                {
                    if (IsReady(*it))
                    {
                        queue.erase(it);
                        break;
                    }
                }
            }

        }
        Vector::Vector(const Vector& input)
        {
            (*this) = input;
        }
        Vector::Vector(const Devices& devices, const eilig::Vector& input)
        {
            SetDevices(devices);
            Resize(input.GetRows());

            std::deque<std::future<bool>> queue;

            for (Index i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i]->GetType() == device_vector_cpu)
                {
                    auto task = std::static_pointer_cast<VectorCPU>(devices_[i])->GetKernelCopyVector(input);
                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            while (!queue.empty())
            {
                const auto& it = queue.begin();

                while (it != queue.end())
                {
                    if (IsReady(*it))
                    {
                        queue.erase(it);
                        break;
                    }
                }
            }
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
        void Vector::Resize(NumberRows numberRows)
        {            
            if (numberRows == 0)
            {
                throw std::invalid_argument("Vector dimensions cannot be zero.");
            }

			numberRows_ = numberRows;

            std::deque<std::future<bool>> queue;

            for (Index i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i]->GetType() == device_vector_cpu)
                {
                    auto task = std::static_pointer_cast<VectorCPU>(devices_[i])->GetKernelResize(numberRows);
                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            while (!queue.empty())
            {
                const auto& it = queue.begin();

                while (it != queue.end())
                {
                    if (IsReady(*it))
                    {
                        queue.erase(it);
                        break;
                    }
                }
            }
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
            return EntryProxyVector(devices_, row);
        }
        Vector& Vector::operator=(Scalar rhs)
        {
            std::deque<std::future<bool>> queue;

            for (Index i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i]->GetType() == device_vector_cpu)
                {
                    auto task = std::static_pointer_cast<VectorCPU>(devices_[i])->GetKernelCopyScalar(rhs);
                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            while (!queue.empty())
            {
                const auto& it = queue.begin();

                while (it != queue.end())
                {
                    if (IsReady(*it))
                    {
                        queue.erase(it);
                        break;
                    }
                }
            }

			return *this;
        }
        Vector& Vector::operator=(const Vector& rhs)
        {
            numberRows_ = rhs.numberRows_;
            devices_ = rhs.devices_;

            return *this;
        }
        Vector& Vector::operator=(Vector&& rhs) noexcept
        {
            if (&rhs == this)
            {
                return *this;
            }

            numberRows_ = rhs.numberRows_;
            devices_ = std::move(rhs.devices_);

            return *this;
        }
        Vector Vector::operator+(Scalar rhs) const
        {
            Vector res(*this);

            std::deque<std::future<bool>> queue;

            for (Index i = 0; i < res.devices_.size(); ++i)
            {
                if (res.devices_[i]->GetType() == device_vector_cpu)
                {
                    auto task = std::static_pointer_cast<VectorCPU>(res.devices_[i])->GetKernelAddScalar(rhs);
                    queue.emplace_back(threadPool.submit_task(task));
                }
            }

            while (!queue.empty())
            {
                const auto& it = queue.begin();

                while (it != queue.end())
                {
                    if (IsReady(*it))
                    {
                        queue.erase(it);
                        break;
                    }
                }
            }

            return res;
        }
        NumberRows Vector::GetRows() const
        {
            return numberRows_;
        }
        NumberCols Vector::GetCols() const
        {
            return 1;
        }
        Scalar Vector::GetValue(Index i) const
        {
            return const_cast<Vector&>(*this)(i)();
        }
        void Vector::SetDevices(const Devices& devices)
        {
            if (devices.size() == 0)
            {
                throw std::invalid_argument("Number of devices cannot be zero.");
            }

            devices_ = devices;
        }
    }
} /* namespace eilig */