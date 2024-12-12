#define EILIG_SCALAR double
#define EILIG_SIZE_T long unsigned int

__kernel void VectorCopyS(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);
	
	if(i < rows)
	{
		y[i] = alpha;		
	}
};
__kernel void VectorAddS(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, __global EILIG_SCALAR* x, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);

	if(i < rows)
	{
		y[i] = x[i] + alpha;		
	}
};
__kernel void VectorAddSl(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, __global EILIG_SCALAR* x, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);
	
	if(i < rows)
	{
		y[i] = alpha + x[i];	
	}
};
__kernel void VectorAddV(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, __global EILIG_SCALAR* x, __global EILIG_SCALAR* w)
{
	size_t i = get_global_id(0);

	if(i < rows)
	{
		y[i] = x[i] + w[i];		
	}	
};
__kernel void VectorPlus(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y)
{
	size_t i = get_global_id(0);

	if(i < rows)
	{
		y[i] = +y[i];
	}
};
__kernel void VectorSubS(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, __global EILIG_SCALAR* x, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);

	if(i < rows)
	{
		y[i] = x[i] - alpha;
	}
};
__kernel void VectorSubSl(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, __global EILIG_SCALAR* x, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);
	
	if(i < rows)
	{
		y[i] = alpha - x[i];
	}
};
__kernel void VectorSubV(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, __global EILIG_SCALAR* x, __global EILIG_SCALAR* w)
{
	size_t i = get_global_id(0);

	if(i < rows)
	{
		y[i] = x[i] - w[i];
	}
};
__kernel void VectorMinus(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y)
{
	size_t i = get_global_id(0);
	
	if(i < rows)
	{
		y[i] = -y[i];
	}
};
__kernel void VectorMulS(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, __global EILIG_SCALAR* x, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);

	if(i < rows)
	{
		y[i] = x[i] * alpha;
	}
};
__kernel void VectorNormMax(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, __global EILIG_SCALAR* partial, __local EILIG_SCALAR* localMem)
{
	size_t global_id = get_global_id(0);
	size_t local_id = get_local_id(0);	
	size_t local_size = get_local_size(0); 
	EILIG_SCALAR partial_max;	
	
	if(global_id < rows)
	{
		localMem[local_id] = fabs(y[global_id]);
	}
	else
	{
		localMem[local_id] = 0.0;		
	}
	work_group_barrier(CLK_LOCAL_MEM_FENCE);	
	
	if(local_id == 0)
	{
		partial_max = localMem[0];
		for(size_t i = 0; i < local_size; i++)
		{
			if(partial_max < localMem[i])
			{
				partial_max = localMem[i];
			}
		}
		
		partial[get_group_id(0)] = partial_max;
	}
};
__kernel void VectorNormP(const EILIG_SIZE_T rows, const EILIG_SCALAR p, __global EILIG_SCALAR* y, __global EILIG_SCALAR* partial, __local EILIG_SCALAR* localMem)
{
	size_t global_id = get_global_id(0);
	size_t local_id = get_local_id(0);	
	size_t local_size = get_local_size(0); 
	EILIG_SCALAR partial_sum;	

	if(global_id < rows)
	{
		localMem[local_id] = pow(fabs(y[global_id]), p);
	}
	else
	{
		localMem[local_id] = 0.0;		
	}	
	work_group_barrier(CLK_LOCAL_MEM_FENCE);	
	
	if(local_id == 0)
	{
		partial_sum = 0.0;
		for(size_t i = 0; i < local_size; i++)
		{
			partial_sum += localMem[i];
		}
		
		partial[get_group_id(0)] = partial_sum;
	}
};
__kernel void VectorNormP2(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, __global EILIG_SCALAR* partial, __local EILIG_SCALAR* localMem)
{
	size_t global_id = get_global_id(0);
	size_t local_id = get_local_id(0);	
	size_t local_size = get_local_size(0); 
	EILIG_SCALAR partial_sum;	

	if(global_id < rows)
	{
		localMem[local_id] = y[global_id] * y[global_id];
	}
	else
	{
		localMem[local_id] = 0.0;		
	}		
	work_group_barrier(CLK_LOCAL_MEM_FENCE);	
	
	if(local_id == 0)
	{
		partial_sum = 0.0;
		for(size_t i = 0; i < local_size; i++)
		{
			partial_sum += localMem[i];
		}
		
		partial[get_group_id(0)] = partial_sum;
	}
};
__kernel void VectorDot(const EILIG_SIZE_T rows, __global EILIG_SCALAR* y, __global EILIG_SCALAR* x, __global EILIG_SCALAR* partial, __local EILIG_SCALAR* localMem)
{
	size_t global_id = get_global_id(0);
	size_t local_id = get_local_id(0);	
	size_t local_size = get_local_size(0); 
	EILIG_SCALAR partial_sum;	

	if(global_id < rows)
	{
		localMem[local_id] = y[global_id] * x[global_id];
	}
	else
	{
		localMem[local_id] = 0.0;		
	}
	work_group_barrier(CLK_LOCAL_MEM_FENCE);	
	
	if(local_id == 0)
	{
		partial_sum = 0.0;
		for(size_t i = 0; i < local_size; i++)
		{
			partial_sum += localMem[i];
		}
		
		partial[get_group_id(0)] = partial_sum;
	}
};
__kernel void EllpackNormP(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, const EILIG_SCALAR p, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, __global EILIG_SCALAR* partial, __local EILIG_SCALAR* localMem)
{
	size_t global_id = get_global_id(0);
	size_t local_id = get_local_id(0);	
	size_t local_size = get_local_size(0); 
	EILIG_SCALAR partial_sum;

	localMem[local_id] = 0.0;

	if(global_id < rows)
	{
		for(size_t i = 0; i < count[global_id]; i++)
		{
			localMem[local_id] += pow(fabs(data[global_id * width + i]), p);
		}
	}
	else
	{
		localMem[local_id] = 0.0;		
	}	
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	
	if(local_id == 0)
	{
		partial_sum = 0.0;
		for(size_t i = 0; i < local_size; i++)
		{
			partial_sum += localMem[i];
		}
		
		partial[get_group_id(0)] = partial_sum;
	}
};
__kernel void EllpackNormP2(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, __global EILIG_SCALAR* partial, __local EILIG_SCALAR* localMem)
{
	size_t global_id = get_global_id(0);
	size_t local_id = get_local_id(0);	
	size_t local_size = get_local_size(0); 
	size_t n = count[global_id];
	EILIG_SCALAR partial_sum;

	localMem[local_id] = 0.0;
	for(size_t i = 0; i < n; i++)
	{
		localMem[local_id] += data[global_id * width + i] * data[global_id * width + i];
	}
	
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	
	if(local_id == 0)
	{
		partial_sum = 0.0;
		for(size_t i = 0; i < local_size; i++)
		{
			partial_sum += localMem[i];
		}
		
		partial[get_group_id(0)] = partial_sum;
	}
};
__kernel void EllpackMaxCount(const EILIG_SIZE_T rows, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* partial, __local EILIG_SIZE_T* localMem)
{
	size_t i = get_global_id(0);
	size_t local_id = get_local_id(0);	
	size_t local_size = get_local_size(0); 
	
	EILIG_SIZE_T partial_max;	

	if(i < rows)
	{
		localMem[local_id] = count[i];
	}
	else
	{
		localMem[local_id] = 0.0;		
	}
	work_group_barrier(CLK_LOCAL_MEM_FENCE);	
	
	if(local_id == 0)
	{
		partial_max = localMem[0];
		for(size_t i = 0; i < local_size; i++)
		{
			if(partial_max < localMem[i])
			{
				partial_max = localMem[i];
			}
		}
		
		partial[get_group_id(0)] = partial_max;
	}
};
__kernel void EllpackExpandPosition(const EILIG_SIZE_T rows, const EILIG_SIZE_T width, const EILIG_SIZE_T expansion, __global EILIG_SIZE_T* y, __global EILIG_SIZE_T* x)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	if( i < rows )
	{
		if(j < width)
		{
			y[i * expansion + j] = x[i * width + j];
		}
		else if(j < expansion)
		{
			y[i * expansion + j] = 0;
		}
	}	
};
__kernel void EllpackExpandData(const EILIG_SIZE_T rows, const EILIG_SIZE_T width, const EILIG_SIZE_T expansion, __global EILIG_SCALAR* y, __global EILIG_SCALAR* x)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	if( i < rows)
	{
		if(j < width)
		{
			y[i * expansion + j] = x[i * width + j];
		}
		else if(j < expansion)
		{
			y[i * expansion + j] = 0.0;
		}
	}	
};
__kernel void EllpackShrinkPosition(const EILIG_SIZE_T rows, const EILIG_SIZE_T width, const EILIG_SIZE_T shrinkage, __global EILIG_SIZE_T* y, __global EILIG_SIZE_T* x)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	if(i < rows)
	{
		if(j < shrinkage)
		{
			y[i * shrinkage + j] = x[i * width + j];
		}
	}
};
__kernel void EllpackShrinkData(const EILIG_SIZE_T rows, const EILIG_SIZE_T width, const EILIG_SIZE_T shrinkage, __global EILIG_SCALAR* y, __global EILIG_SCALAR* x)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	if(i < rows)
	{
		if(j < shrinkage)
		{
			y[i * shrinkage + j] = x[i * width + j];
		}
	}
};
__kernel void EllpackCopyS(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t index;
	
	if(i < rows && j < width)
	{
		if(j == 0 )
		{
			count[i] = cols;
		}
	
		position[i * width + j] = j;
		data[i * width + j] = alpha;	
	}
};
__kernel void EllpackAddS(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t n;
	size_t k;

	if(i < rows && j < cols)
	{
		n = count[i];		
		
		if(j < n)
		{
			k = position[i * width + j];
			dataRes[i * cols + k] += data[i * width + j];
		}	
	}
};
__kernel void EllpackAddSl(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t n;
	size_t k;

	if(i < rows && j < width)
	{
		n = count[i];	
		
		if(j < n)
		{
			k = position[i * width + j];
			dataRes[i * cols + k] += data[i * width + j];
		}	
	}	
};
__kernel void EllpackPlus(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SCALAR* data)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t n;	

	if(i < rows && j < width)
	{
		n = count[i];
		
		if(j < n)
		{
			data[i * width + j] = +data[i * width + j];
		}
	}		
};
__kernel void EllpackSubS(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t n;
	size_t k;

	if(i < rows && j < width)
	{
		n = count[i];
		
		if(j < n)
		{
			k = position[i * width + j];			
			dataRes[i * cols + k] += data[i * width + j];
		}
	}
};
__kernel void EllpackSubSl(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t n;
	size_t k;

	if(i < rows && j < width)
	{
		n = count[i];	
		
		if(j < n)
		{
			k = position[i * width + j];
			dataRes[i * cols + k] -= data[i * width + j];
		}	
	}	
};
__kernel void EllpackMinus(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SCALAR* data)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t n;	

	if(i < rows && j < width)
	{
		n = count[i];
		
		if(j < n)
		{
			data[i * width + j] = -data[i * width + j];
		}
	}		
};
__kernel void EllpackMulS(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes, const EILIG_SCALAR alpha)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t n;

	if(i < rows && j < width)
	{
		n = count[i];
		
		if(j < n)
		{
			dataRes[i * width + j] = alpha * data[i * width + j];
		}
	}	
};
__kernel void EllpackMulV(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, __global EILIG_SCALAR* vector, __global EILIG_SCALAR* dataRes)
{
	size_t i = get_global_id(0);
	size_t n;
	size_t k;
	
	EILIG_SCALAR sum = 0.0;

	if(i < rows)
	{
		n = count[i];
		
		for(size_t j = 0; j < n; j++ )
		{
			k = position[i * width + j];
			sum += data[i * width + j] * vector[k];
		}

		dataRes[i] = sum;
	}
};
__kernel void EllpackMulM(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SIZE_T rowsTrans, const EILIG_SIZE_T colsTrans, const EILIG_SIZE_T widthTrans, __global EILIG_SIZE_T* countTrans, __global EILIG_SIZE_T* positionTrans, __global EILIG_SCALAR* dataTrans, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes, __local size_t* localMem)
{
	size_t i = get_global_id(0);
	size_t n = count[i];
	size_t col;
	
	EILIG_SCALAR sum;
	
	for(size_t j = 0; j < rowsTrans; j++ )
	{
		if( i == 0 )
		{
			for(size_t k = 0; k < cols; k++ )
			{
				localMem[k] = 0.0;				
			}
			
			for(size_t k = 0; k < widthTrans; k++ )
			{		
				col = positionTrans[ j * widthTrans + k];
				localMem[col] = dataTrans[ j * widthTrans + k ];
			}
		}		
		
		barrier(CLK_LOCAL_MEM_FENCE); 		
		
		sum = 0.0;
		for(size_t k = 0; k < n; k++ )
		{
			col = position[i * width + k];
			sum += data[i * width + k] * localMem[col];
		}
		
		if( sum != 0.0 )
		{
			positionRes[i * widthTrans + countRes[i]] = j;
			dataRes[i * widthTrans + countRes[i]] = sum;
			countRes[i] += 1;
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); 	
	}			
};
__kernel void EllpackSwapRows(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SIZE_T row1, const EILIG_SIZE_T row2)
{
	size_t i = get_global_id(0);
	
	size_t countT;
	size_t positionT;
	EILIG_SCALAR dataT;

	if(i < width)
	{
		positionT = position[row1 * width + i];
		position[row1 * width + i] = position[row2 * width + i];
		position[row2 * width + i] = positionT;

		dataT = data[row1 * width + i];
		data[row1 * width + i] = data[row2 * width + i];
		data[row2 * width + i] = dataT;
		
		if(i == 0)
		{
			countT = count[row1];
			count[row1] = count[row2];
			count[row2] = countT;
		}		
	}
};
__kernel void EllpackSwapCols(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SIZE_T col1, const EILIG_SIZE_T col2)
{
	size_t i = get_global_id(0);
	size_t n;
	
	bool found1 = false;
	bool found2 = false;
	
	size_t index1;
	size_t index2;
	size_t indexT;
	size_t positionT;
	EILIG_SCALAR dataT;
	
	if(i < rows)
	{
		n = count[i];
		
		for( size_t j = 0; j < n; j++ )
		{
			if( position[i * width + j] == col1 )
			{
				found1 = true;
				index1 = j;
			}
			
			if( position[i * width + j] == col2 )
			{
				found2 = true;
				index2 = j;
			}		
		}

		if( found1 && !found2 )
		{
			position[i * width + index1] = col2;		

			for( size_t j = 0; j < (n - 1); j++ )
			{
				for( size_t k = 0; k < (n - j - 1); k++ )
				{
					if( position[i * width + k] >  position[i * width + k + 1])
					{	
						positionT = position[i * width + k];
						position[i * width + k] = position[i * width + k + 1];
						position[i * width + k + 1] = positionT;

						dataT = data[i * width + k];
						data[i * width + k] = data[i * width + k + 1];
						data[i * width + k + 1] = dataT;
					}
				}
			}					
		}
		else if( !found1 && found2 )
		{
			position[i * width + index2] = col1;		

			for( size_t j = 0; j < (n - 1); j++ )
			{
				for( size_t k = 0; k < (n - j - 1); k++ )
				{
					if( position[i * width + k] >  position[i * width + k + 1])
					{	
						positionT = position[i * width + k];
						position[i * width + k] = position[i * width + k + 1];
						position[i * width + k + 1] = positionT;

						dataT = data[i * width + k];
						data[i * width + k] = data[i * width + k + 1];
						data[i * width + k + 1] = dataT;
					}
				}
			}			
		}
		else if( found1 && found2 )
		{
			positionT = position[i * width + index1];
			position[i * width + index1]  = position[i * width + index2];
			position[i * width + index2]  = positionT;

			for( size_t j = 0; j < (n - 1); j++ )
			{
				for( size_t k = 0; k < (n - j - 1); k++ )
				{
					if( position[i * width + k] >  position[i * width + k + 1])
					{	
						positionT = position[i * width + k];
						position[i * width + k] = position[i * width + k + 1];
						position[i * width + k + 1] = positionT;

						dataT = data[i * width + k];
						data[i * width + k] = data[i * width + k + 1];
						data[i * width + k + 1] = dataT;
					}
				}
			}		
		}		
	}
};
__kernel void EllpackTranspose(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SIZE_T widthRes, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes)
{
	size_t j = get_global_id(0);
	size_t n;

	if(j < cols)
	{
		for( size_t i = 0; i < rows; i++ )
		{			
			n = count[i];
			
			for( size_t k = 0; k < n; k++ )
			{			
				if(position[ i * width + k ] == j)
				{
					positionRes[j * widthRes + countRes[j]] = i;
					dataRes[j * widthRes + countRes[j]] = data[i * width + k];
					countRes[j] += 1;
					break;
				}		
			}		
		}	
	}
};
__kernel void EllpackFindWidthTranspose(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, volatile __global EILIG_SIZE_T* countRes)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);	
	size_t n;

	if(i < rows && j < cols)
	{
		n = count[i];
		
		for( size_t k = 0; k < n; k++ )
		{			
			if(position[i * width + k] == j)
			{
				atom_add(&countRes[j], 1);
				break;
			}		
		}
	}
};
__kernel void EllpackDiagonal(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SIZE_T widthRes, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes)
{
	size_t i = get_global_id(0);
	size_t n;	
	
	if(i < rows)
	{
		n = count[i];
		
		for( size_t j = 0; j < n; j++ )
		{
			if(position[ i * width + j] == i)
			{
				countRes[i] = 1;
				positionRes[ i * widthRes + 0] = i;
				dataRes[ i * widthRes + 0] = data[ i * width + j];
				break;
			}
			
			if(position[ i * width + j] > i)
			{
				break;
			}
		}	
	}
};
__kernel void EllpackRegion(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SIZE_T widthRes, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes, const EILIG_SIZE_T aux1, const EILIG_SIZE_T aux2, const EILIG_SIZE_T aux3, const EILIG_SIZE_T aux4)
{
	size_t i = get_global_id(0);
	size_t n;	
	size_t col = 0;

	if(i < aux1)
	{
		n = count[aux3 + i];	
			
		for( size_t j = 0; j < n; j++ )
		{
			col = position[(aux3 + i) * width + j];

			if ( (col >= aux4) && (col < (aux4 + aux2)) )
			{				
				positionRes[i * widthRes + countRes[i]] = col - aux4;
				dataRes[i * widthRes + countRes[i]] = data[(aux3 + i) * width + j];
				countRes[i] += 1;
			}
		}		
	}	
};
__kernel void EllpackLower1(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SIZE_T widthRes, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes)
{
	size_t i = get_global_id(0);
	size_t n;	
	size_t pos = 0;	

	if(i < rows)
	{
		n = count[i];
		countRes[i] = 0;	
		
		for( size_t j = 0; j < n; j++ )
		{
			if(position[ i * width + j] <= i)
			{
				pos = countRes[i];
				positionRes[ i * widthRes + pos] = position[ i * width + j];
				dataRes[ i * widthRes + pos] = data[ i * width + j];
				pos += 1;
				countRes[i] = pos;
			}
		}
	}		
};
__kernel void EllpackLower2(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SIZE_T widthRes, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes)
{
	size_t i = get_global_id(0);
	size_t n;	
	size_t pos = 0;	

	if(i < rows)
	{
		n = count[i];
		countRes[i] = 0;
		for( size_t j = 0; j < n; j++ )
		{
			if(position[ i * width + j] < i)
			{
				pos = countRes[i];
				positionRes[ i * widthRes + pos] = position[ i * width + j];
				dataRes[ i * widthRes + pos] = data[ i * width + j];
				pos += 1;
				countRes[i] = pos;			
			}
		}
	}
	
			
};
__kernel void EllpackUpper1(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SIZE_T widthRes, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes)
{
	size_t i = get_global_id(0);
	size_t n;	
	size_t pos = 0;	

	if(i < rows)
	{
		n = count[i];
		countRes[i] = 0;
		
		for( size_t j = 0; j < n; j++ )
		{
			if(position[ i * width + j] >= i)
			{
				pos = countRes[i];
				positionRes[ i * widthRes + pos] = position[ i * width + j];
				dataRes[ i * widthRes + pos] = data[ i * width + j];
				pos += 1;
				countRes[i] = pos;			
			}
		}
	}	
};
__kernel void EllpackUpper2(const EILIG_SIZE_T rows, const EILIG_SIZE_T cols, const EILIG_SIZE_T width, __global EILIG_SIZE_T* count, __global EILIG_SIZE_T* position, __global EILIG_SCALAR* data, const EILIG_SIZE_T widthRes, __global EILIG_SIZE_T* countRes, __global EILIG_SIZE_T* positionRes, __global EILIG_SCALAR* dataRes)
{
	size_t i = get_global_id(0);
	size_t n;	
	size_t pos = 0;	

	if(i < rows)
	{
		n = count[i];
		countRes[i] = 0;
		
		for( size_t j = 0; j < n; j++ )
		{
			if(position[ i * width + j] > i)
			{
				pos = countRes[i];
				positionRes[ i * widthRes + pos] = position[ i * width + j];
				dataRes[ i * widthRes + pos] = data[ i * width + j];
				pos += 1;
				countRes[i] = pos;			
			}
		}	
	}
};
