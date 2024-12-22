%module eilig

%{
	#define SWIG_PYTHON_EXTRA_NATIVE_CONTAINERS 
%}

%include "typemaps.i"
%include "cpointer.i"
%include "std_shared_ptr.i"
%include "std_string.i"
%include "std_set.i"
%include "std_vector.i"

%inline
%{
	#include "eilig_types.hpp"
	#include "eilig_vector.hpp"
	#include "eilig_matrix.hpp"
	#include "eilig_matrix_ellpack.hpp"
	#include "eilig_routines.hpp"
	#include "eilig_transform.hpp"

	using Scalar = double;
	using Status = long long int;
	using NumberRows = std::size_t;
	using NumberCols = std::size_t;
	using String = std::string;
	using Index = std::size_t;
	using Axis = std::size_t;
	
	using Indices = std::vector<Index>;
	using Scalars = std::vector<Scalar>;
%}

namespace std 
{
	typedef ::size_t size_t;
}

%template(vecSizeT) std::vector<std::size_t>;
%template(vecDouble) std::vector<double>;
%template(arrDouble) std::vector<std::vector<double>>;

%include "eilig_types.hpp"
%include "eilig_vector.hpp"
%include "eilig_matrix.hpp"
%include "eilig_matrix_ellpack.hpp"
%include "eilig_routines.hpp"
%include "eilig_transform.hpp"

%extend eilig::Vector {
    String __str__() const 
	{
        return eilig::ListVector(*self);
    }
	
	Vector __radd__(Scalar value) const 
	{
        return value + (*self);
    }
	
	Vector __rsub__(Scalar value) const 
	{
        return value - (*self);
    }	
	
	Vector __rmul__(Scalar value) const 
	{
        return value * (*self);
    }	
}

%extend eilig::Matrix {
    String __str__() const 
	{
        return eilig::ListMatrix(*self);
    }
	
	Matrix __radd__(Scalar value) const 
	{
        return value + (*self);
    }
	
	Matrix __rsub__(Scalar value) const 
	{
        return value - (*self);
    }	
	
	Matrix __rmul__(Scalar value) const 
	{
        return value * (*self);
    }		
}

%extend eilig::Ellpack {
    String __str__() const 
	{
        return eilig::ListMatrix(*self);
    }
	
	Ellpack __radd__(Scalar value) const 
	{
        return value + (*self);
    }
	
	Ellpack __rsub__(Scalar value) const 
	{
        return value - (*self);
    }	
	
	Ellpack __rmul__(Scalar value) const 
	{
        return value * (*self);
    }		
}

%pythoncode
%{

def GetItemVector(self, index):
    return self.GetValue(index)

def SetItemVector(self, index, value):
    return self.SetValue(index, value)

Vector.__getitem__ = GetItemVector
Vector.__setitem__ = SetItemVector

def GetItemMatrix(self, index):
    return self.GetValue(index[0], index[1])

def SetItemMatrix(self, index, value):
    return self.SetValue(index[0], index[1], value)

Matrix.__getitem__ = GetItemMatrix
Matrix.__setitem__ = SetItemMatrix

def GetItemEllpack(self, index):
    return self.GetValue(index[0], index[1])

def SetItemEllpack(self, index, value):
    return self.SetValue(index[0], index[1], value)

Ellpack.__getitem__ = GetItemEllpack
Ellpack.__setitem__ = SetItemEllpack

%}