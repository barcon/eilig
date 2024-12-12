import os
import random

#os.add_dll_directory(r'Z:\devel\eilig\test')

import eilig

platformNumber = 1
deviceNumber = 0
rows = 3201

v1_FL = eilig.Vector(rows)
v2_FL = eilig.Vector(rows)

kernels = eilig.CreateKernels("kernels.c", platformNumber, deviceNumber);

v1_CL = eilig.VectorCL(kernels, rows)
v2_CL = eilig.VectorCL(kernels, rows)

for i in range (0, rows):
    value = random.random()
    v1_FL.SetValue(i, value)
    v1_CL.SetValue(i, value)
    
    value = random.random()
    v2_FL.SetValue(i, value)
    v2_CL.SetValue(i, value)     

#------TEST 1 - Get Rows and Cols----------------------------------------------
res1 = v1_FL.GetRows()
res2 = v1_CL.GetRows()

res3 = v1_FL.GetCols()
res4 = v1_CL.GetCols()

if( abs(res1 - res2) == 0 and abs(res3 - res4) == 0):
    print("Test (001) - Vector Get Rows and Cols : PASSED")
else:   
    print("Test (001) - Vector Get Rows and Cols : NOT PASSED") 
 
#------TEST 2 - Norm Max-------------------------------------------------------
res1 = eilig.NormMax(v1_FL)
res2 = eilig.NormMax(v1_CL)

if( (abs(res1 - res2) / abs(res1)) < 10**(- 6)):
    print("Test (002) - Vector Norm Max : PASSED")
else:   
    print("Test (002) - Vector Norm Max : NOT PASSED")

#------TEST 3 - Norm P---------------------------------------------------------
res1 = eilig.NormP(v1_FL, 2)
res2 = eilig.NormP(v1_CL, 2)

if( (abs(res1 - res2) / abs(res1)) < 10**(- 6)):
    print("Test (003) - Vector Norm P : PASSED")
else:   
    print("Test (003) - Vector Norm P : NOT PASSED")

#------TEST 4 - Norm P2--------------------------------------------------------
res1 = eilig.NormP2(v1_FL)
res2 = eilig.NormP2(v1_CL)

if( (abs(res1 - res2) / abs(res1)) < 10**(- 6)):
    print("Test (004) - Vector Norm P2 : PASSED")
else:   
    print("Test (004) - Vector Norm P2 : NOT PASSED")
    
#------TEST 5 - Dot Product----------------------------------------------------
res1 = eilig.Dot(v1_FL, v2_FL)
res2 = eilig.Dot(v1_CL, v2_CL)

if( (abs(res1 - res2) / abs(res1)) < 10**(- 6)):
    print("Test (005) - Vector Dot Product : PASSED")
else:   
    print("Test (005) - Vector Dot Product : NOT PASSED")

#------TEST 6 - Vector Add-----------------------------------------------------
res1 = eilig.NormP2(v1_FL + v2_FL)
res2 = eilig.NormP2(v1_CL + v2_CL)

if( (abs(res1 - res2) / abs(res1)) < 10**(- 6)):
    print("Test (006) - Vector Add : PASSED")
else:   
    print("Test (006) - Vector Add : NOT PASSED")

#------TEST 7 - Vector Sub-----------------------------------------------------
res1 = eilig.NormP2(v1_FL - v2_FL)
res2 = eilig.NormP2(v1_CL - v2_CL)

if( (abs(res1 - res2) / abs(res1)) < 10**(- 6)):
    print("Test (007) - Vector Sub : PASSED")
else:   
    print("Test (007) - Vector Sub : NOT PASSED")

#------TEST 8 - Vector Scalar Multiply-----------------------------------------
res1 = eilig.NormP2(v1_FL * 2.0)
res2 = eilig.NormP2(v1_CL * 2.0)

if( (abs(res1 - res2) / abs(res1)) < 10**(- 6)):
    print("Test (008) - Vector Scalar Multiply : PASSED")
else:   
    print("Test (008) - Vector Scalar Multiply : NOT PASSED")