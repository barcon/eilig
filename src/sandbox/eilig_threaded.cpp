#include "eilig_threaded.hpp"
#include "eilig_routines.hpp"

namespace eilig
{
    namespace threaded
    {
        NumberProcessors numberProcessors = std::thread::hardware_concurrency() > 1 ? std::thread::hardware_concurrency() : 1;        
        ThreadPool threadPool(numberProcessors);

		void AdjustBlock(Block& block, NumberRows numberRows, Number numberDevices, Index index)
		{
			NumberRows blockSize{0};
			NumberRows restSize {0};

			if (numberRows < numberDevices)
			{
				blockSize = 1;
				restSize = 0;

				if (index > numberRows - 1)
				{
					block.isUsed = false;
					block.row = 0;
					block.numberRows = 0;
				}
				else
				{
					block.isUsed = true;
					block.row = index * blockSize;
					block.numberRows = 1;
				}
			}
			else
			{
				blockSize = numberRows / numberDevices;
				restSize = numberRows % numberDevices;

				block.isUsed = true;
				block.row = index * blockSize;

				if (index == numberDevices - 1)
				{
					block.numberRows = blockSize + restSize;
				}
				else
				{
					block.numberRows = blockSize;
				}
			}
		}
		Offset GetOffset(const Blocks& blocks, Index row)
		{
			Offset offset{ 0, 0 };

			for (Index i = 0; i < blocks.size(); ++i)
			{
				if(!blocks[i].isUsed)
				{
					continue;
				}

				if (row >= blocks[i].row && row < blocks[i].row + blocks[i].numberRows)
				{
					offset.block = i;
					break;
				}
			}

			offset.row = row - blocks[offset.block].row;

			return offset;
		}
		
		Devices GetDevices(const Blocks& blocks)
		{
			Devices res;
			
			for(auto& block : blocks)
			{
				res.emplace_back(block.device);
			}

			return res;
		}

		void WaitForAll(TaskQueueBool& queue)
		{
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
		void WaitForAll_Ellpack_Vector_Multiplication(TaskQueueVector& queue, TaskQueueVectorCL& queueCL, Vector& result)
		{
			while (!queue.empty())
			{
				const auto& it = queue.begin();

				while (it != queue.end())
				{
					if (IsReady(*it))
					{
						auto taskBlock = it->get();
						result.Replace(taskBlock.block.row, taskBlock.value);
						
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
						result.Replace(taskBlock.block.row, taskBlock.value.Convert());

						queueCL.erase(it);
						break;
					}
				}
			}
		}
		Scalar WaitForAll_Ellpack_Sum(TaskQueueScalar& queue)
		{
			Scalar res{ 0.0 };

			while (!queue.empty())
			{
				const auto& it = queue.begin();

				while (it != queue.end())
				{
					if (IsReady(*it))
					{
						res += it->get();

						queue.erase(it);
						break;
					}
				}
			}

			return res;
		}
		Ellpack WaitForAll_Ellpack_Diagonal(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL)
		{
			auto taskBlocks = std::vector<TaskBlock<eilig::Ellpack>>();
			auto taskBlocksCL = std::vector<TaskBlock<eilig::opencl::Ellpack>>();

			NumberRows numberRows{ 0 };
			NumberCols numberCols{ 0 };
			Devices devices;

			devices.resize(queue.size() + queueCL.size());

			while (!queue.empty())
			{
				const auto& it = queue.begin();

				while (it != queue.end())
				{
					if (IsReady(*it))
					{
						auto taskBlock = it->get();
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;

						taskBlocks.emplace_back(taskBlock);
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
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;
						taskBlocksCL.emplace_back(taskBlock);
						queueCL.erase(it);
						break;
					}
				}
			}

			eilig::Ellpack aux(numberRows, numberCols);

			for (Index i = 0; i < taskBlocks.size(); i++)
			{
				auto& block = taskBlocks[i].block;

				aux.Replace(block.row, 0, taskBlocks[i].value);
			}

			for (Index i = 0; i < taskBlocksCL.size(); i++)
			{
				auto& block = taskBlocksCL[i].block;

				aux.Replace(block.row, 0, taskBlocksCL[i].value.Convert());
			}

			return Ellpack(devices, aux);
		}
		Ellpack WaitForAll_Ellpack_Diagonal_Scale(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL)
		{
			auto taskBlocks = std::vector<TaskBlock<eilig::Ellpack>>();
			auto taskBlocksCL = std::vector<TaskBlock<eilig::opencl::Ellpack>>();

			NumberRows numberRows{ 0 };
			NumberCols numberCols{ 0 };
			Devices devices;

			devices.resize(queue.size() + queueCL.size());

			while (!queue.empty())
			{
				const auto& it = queue.begin();

				while (it != queue.end())
				{
					if (IsReady(*it))
					{
						auto taskBlock = it->get();
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;
						taskBlocks.emplace_back(taskBlock);
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
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;
						taskBlocksCL.emplace_back(taskBlock);
						queueCL.erase(it);
						break;
					}
				}
			}

			eilig::Ellpack aux(numberRows, numberCols);

			for (Index i = 0; i < taskBlocks.size(); i++)
			{
				auto& block = taskBlocks[i].block;

				aux.Replace(block.row, 0, taskBlocks[i].value);
			}

			for (Index i = 0; i < taskBlocksCL.size(); i++)
			{
				auto& block = taskBlocksCL[i].block;

				aux.Replace(block.row, 0, taskBlocksCL[i].value.Convert());
			}

			return Ellpack(devices, aux);
		}
		Vector	WaitForAll_Ellpack_Diagonal_Vector(TaskQueueVector& queue, TaskQueueVectorCL& queueCL)
		{
			auto taskBlocks = std::vector<TaskBlock<eilig::Vector>>();
			auto taskBlocksCL = std::vector<TaskBlock<eilig::opencl::Vector>>();

			NumberRows numberRows{ 0 };
			Devices devices;

			devices.resize(queue.size() + queueCL.size());

			while (!queue.empty())
			{
				const auto& it = queue.begin();

				while (it != queue.end())
				{
					if (IsReady(*it))
					{
						auto taskBlock = it->get();
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						devices[index] = taskBlock.block.device;

						taskBlocks.emplace_back(taskBlock);
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
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						devices[index] = taskBlock.block.device;

						taskBlocksCL.emplace_back(taskBlock);
						queueCL.erase(it);
						break;
					}
				}
			}

			eilig::Vector aux(numberRows);

			for (Index i = 0; i < taskBlocks.size(); i++)
			{
				auto& block = taskBlocks[i].block;

				aux.Replace(block.row, taskBlocks[i].value);
			}

			for (Index i = 0; i < taskBlocksCL.size(); i++)
			{
				auto& block = taskBlocksCL[i].block;

				aux.Replace(block.row, taskBlocksCL[i].value.Convert());
			}

			return Vector(devices, aux);
		}
		Ellpack WaitForAll_Ellpack_Lower_With_Diagonal(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL)
		{
			auto taskBlocks = std::vector<TaskBlock<eilig::Ellpack>>();
			auto taskBlocksCL = std::vector<TaskBlock<eilig::opencl::Ellpack>>();

			NumberRows numberRows{ 0 };
			NumberCols numberCols{ 0 };
			Devices devices;

			devices.resize(queue.size() + queueCL.size());

			while (!queue.empty())
			{
				const auto& it = queue.begin();

				while (it != queue.end())
				{
					if (IsReady(*it))
					{
						auto taskBlock = it->get();
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;
						taskBlocks.emplace_back(taskBlock);
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
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;
						taskBlocksCL.emplace_back(taskBlock);
						queueCL.erase(it);
						break;
					}
				}
			}

			eilig::Ellpack aux(numberRows, numberCols);

			for (Index i = 0; i < taskBlocks.size(); i++)
			{
				auto& block = taskBlocks[i].block;

				aux.Replace(block.row, 0, taskBlocks[i].value);
			}

			for (Index i = 0; i < taskBlocksCL.size(); i++)
			{
				auto& block = taskBlocksCL[i].block;

				aux.Replace(block.row, 0, taskBlocksCL[i].value.Convert());
			}

			return Ellpack(devices, aux);
		}
		Ellpack WaitForAll_Ellpack_Lower_Without_Diagonal(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL)
		{
			auto taskBlocks = std::vector<TaskBlock<eilig::Ellpack>>();
			auto taskBlocksCL = std::vector<TaskBlock<eilig::opencl::Ellpack>>();

			NumberRows numberRows{ 0 };
			NumberCols numberCols{ 0 };
			Devices devices;

			devices.resize(queue.size() + queueCL.size());

			while (!queue.empty())
			{
				const auto& it = queue.begin();

				while (it != queue.end())
				{
					if (IsReady(*it))
					{
						auto taskBlock = it->get();
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;
						taskBlocks.emplace_back(taskBlock);
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
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;

						taskBlocksCL.emplace_back(taskBlock);
						queueCL.erase(it);
						break;
					}
				}
			}

			eilig::Ellpack aux(numberRows, numberCols);

			for (Index i = 0; i < taskBlocks.size(); i++)
			{
				auto& block = taskBlocks[i].block;

				aux.Replace(block.row, 0, taskBlocks[i].value);
			}

			for (Index i = 0; i < taskBlocksCL.size(); i++)
			{
				auto& block = taskBlocksCL[i].block;

				aux.Replace(block.row, 0, taskBlocksCL[i].value.Convert());
			}

			return Ellpack(devices, aux);
		}
		Ellpack WaitForAll_Ellpack_Upper_With_Diagonal(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL)
		{
			auto taskBlocks = std::vector<TaskBlock<eilig::Ellpack>>();
			auto taskBlocksCL = std::vector<TaskBlock<eilig::opencl::Ellpack>>();

			NumberRows numberRows{ 0 };
			NumberCols numberCols{ 0 };
			Devices devices;

			devices.resize(queue.size() + queueCL.size());

			while (!queue.empty())
			{
				const auto& it = queue.begin();

				while (it != queue.end())
				{
					if (IsReady(*it))
					{
						auto taskBlock = it->get();
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;
						taskBlocks.emplace_back(taskBlock);
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
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;
						taskBlocksCL.emplace_back(taskBlock);
						queueCL.erase(it);
						break;
					}
				}
			}

			eilig::Ellpack aux(numberRows, numberCols);

			for (Index i = 0; i < taskBlocks.size(); i++)
			{
				auto& block = taskBlocks[i].block;

				aux.Replace(block.row, 0, taskBlocks[i].value);
			}

			for (Index i = 0; i < taskBlocksCL.size(); i++)
			{
				auto& block = taskBlocksCL[i].block;

				aux.Replace(block.row, 0, taskBlocksCL[i].value.Convert());
			}

			return Ellpack(devices, aux);
		}
		Ellpack WaitForAll_Ellpack_Upper_Without_Diagonal(TaskQueueEllpack& queue, TaskQueueEllpackCL& queueCL)
		{
			auto taskBlocks = std::vector<TaskBlock<eilig::Ellpack>>();
			auto taskBlocksCL = std::vector<TaskBlock<eilig::opencl::Ellpack>>();

			NumberRows numberRows{ 0 };
			NumberCols numberCols{ 0 };
			Devices devices;

			devices.resize(queue.size() + queueCL.size());

			while (!queue.empty())
			{
				const auto& it = queue.begin();

				while (it != queue.end())
				{
					if (IsReady(*it))
					{
						auto taskBlock = it->get();
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;
						taskBlocks.emplace_back(taskBlock);
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
						auto index = taskBlock.block.index;

						numberRows += taskBlock.block.numberRows;
						numberCols = taskBlock.value.GetCols();
						devices[index] = taskBlock.block.device;

						taskBlocksCL.emplace_back(taskBlock);
						queueCL.erase(it);
						break;
					}
				}
			}

			eilig::Ellpack aux(numberRows, numberCols);

			for (Index i = 0; i < taskBlocks.size(); i++)
			{
				auto& block = taskBlocks[i].block;

				aux.Replace(block.row, 0, taskBlocks[i].value);
			}

			for (Index i = 0; i < taskBlocksCL.size(); i++)
			{
				auto& block = taskBlocksCL[i].block;

				aux.Replace(block.row, 0, taskBlocksCL[i].value.Convert());
			}

			return Ellpack(devices, aux);
		}

	}
} /* namespace eilig */