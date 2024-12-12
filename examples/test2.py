import math
import random
import eilig

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return math.fabs(a-b) <= max(rel_tol * max(math.fabs(a), math.fabs(b)), abs_tol)

platformNumber = 1
deviceNumber = 0
rows = 50
cols = 50

kernels = eilig.CreateKernels("kernels.c", platformNumber, deviceNumber);

v1_FL = eilig.Vector(rows)
v2_FL = eilig.Vector(cols)
m1_FL = eilig.Matrix(rows, cols)
m2_FL = eilig.Matrix(rows, cols)

for i in range (0, rows):  
    value = random.random()
    m = random.randint(0, rows - 1)    
    v1_FL.SetValue(m, value)

for i in range (0, cols):  
    value = random.random()
    m = random.randint(0, cols - 1)    
    v2_FL.SetValue(m, value)    

for i in range (0, int( rows * cols)):  
    value = random.random()
    m = random.randint(0, rows - 1)
    n = random.randint(0, cols - 1)
    m1_FL.SetValue(m, n, value)   

for i in range (0, int( rows * cols)):  
    value = random.random()
    m = random.randint(0, rows - 1)
    n = random.randint(0, cols - 1)
    m2_FL.SetValue(m, n, value)     

v1_EL = eilig.Vector(v1_FL)        
v2_EL = eilig.Vector(v2_FL)        
m1_EL = eilig.Ellpack(m1_FL)
m2_EL = eilig.Ellpack(m2_FL)

v1_CL = eilig.VectorCL(kernels, v1_FL)        
v2_CL = eilig.VectorCL(kernels, v2_FL)        
m1_CL = eilig.EllpackCL(kernels, m1_FL)
m2_CL = eilig.EllpackCL(kernels, m2_FL)

#------TEST - Matrix Init-----------------------------------------------------
res1 = m1_FL
res2 = m1_CL
passed = True

for i in range (0, rows): 
    for j in range (0, cols):  
        if(res1.GetValue(i, j) != res2.GetValue(i, j)):
            passed = False
            break
    
    if(passed == False):
        break
    
if(passed):
    print("Test - Matrix CL Init: PASSED")
else:
    print("Test - Matrix CL Init: NOT PASSED")

res1 = m2_FL
res2 = m2_CL
passed = True

for i in range (0, rows): 
    for j in range (0, cols):  
        if(res1.GetValue(i, j) != res2.GetValue(i, j)):
            passed = False
            break
    
    if(passed == False):
        break
    
if(passed):
    print("Test - Matrix CL Init: PASSED")
else:
    print("Test - Matrix CL Init: NOT PASSED")

res1 = m1_FL
res2 = m1_EL
passed = True

for i in range (0, rows): 
    for j in range (0, cols):  
        if(res1.GetValue(i, j) != res2.GetValue(i, j)):
            passed = False
            break
    
    if(passed == True):
        break
   
if(passed):
    print("Test - Matrix EL Init: PASSED")
else:
    print("Test - Matrix EL Init: NOT PASSED")
    
res1 = m2_FL
res2 = m2_EL
passed = True

for i in range (0, rows): 
    for j in range (0, cols):  
        if(res1.GetValue(i, j) != res2.GetValue(i, j)):
            passed = False
            break
    
    if(passed == True):
        break
   
if(passed):
    print("Test - Matrix EL Init: PASSED")
else:
    print("Test - Matrix EL Init: NOT PASSED")    

#------TEST - Get Rows --------------------------------------------------------
res1 = m1_FL.GetRows()
res2 = m1_CL.GetRows()
res3 = m1_EL.GetRows()

if(abs(res1 - res2) == 0):
    print("Test - Matrix CL Get Rows: PASSED")
else:   
    print("Test - Matrix CL Get Rows: NOT PASSED")        

if(abs(res1 - res3) == 0):
    print("Test - Matrix EL Get Rows: PASSED")
else:   
    print("Test - Matrix EL Get Rows: NOT PASSED")        

#------TEST - Get Cols --------------------------------------------------------
res1 = m1_FL.GetCols()
res2 = m1_CL.GetCols()
res3 = m1_EL.GetCols()

if(abs(res1 - res2) == 0):
    print("Test - Matrix CL Get Cols: PASSED")
else:   
    print("Test - Matrix CL Get Cols: NOT PASSED")        

if(abs(res1 - res3) == 0):
    print("Test - Matrix EL Get Cols: PASSED")
else:   
    print("Test - Matrix EL Get Cols: NOT PASSED")   

#------TEST - Matrix Is Used-------------------------------------------------------------
res1 = m1_FL
res2 = m1_EL
flag = False

for i in range (0, rows): 
    for j in range (0, cols):  
        if(res1.GetValue(i, j) == 0.0):
            if(res2.IsUsed(i, j) == True):
                flag = True
                break
        else:
            if(res2.IsUsed(i, j) == False):
                flag = True
                break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix EL IsUsed: PASSED")
else:
    print("Test - Matrix EL IsUsed: NOT PASSED")

res1 = m1_FL
res2 = m1_CL
flag = False

for i in range (0, rows): 
    for j in range (0, cols):  
        if(res1.GetValue(i, j) == 0.0):
            if(res2.IsUsed(i, j) == True):
                flag = True
                break
        else:
            if(res2.IsUsed(i, j) == False):
                flag = True
                break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix CL IsUsed: PASSED")
else:
    print("Test - Matrix CL IsUsed: NOT PASSED")

#------TEST - Matrix Norm P2 --------------------------------------------------
res1 = eilig.NormP2(m1_FL)
res2 = eilig.NormP2(m1_CL)
res3 = eilig.NormP2(m1_EL)

if(isclose(res1, res2)):
    print("Test - Matrix CL Norm P2: PASSED")
else:
    print("Test - Matrix CL Norm P2: NOT PASSED")
    
if(isclose(res1, res3)):
    print("Test - Matrix EL Norm P2: PASSED")
else:
    print("Test - Matrix EL Norm P2: NOT PASSED")       

#------TEST - Matrix Norm P ---------------------------------------------------
res1 = eilig.NormP(m1_FL, 0.25)
res2 = eilig.NormP(m1_CL, 0.25)
res3 = eilig.NormP(m1_EL, 0.25)

if(isclose(res1, res2)):
    print("Test - Matrix CL Norm P: PASSED")
else:
    print("Test - Matrix CL Norm P: NOT PASSED")
    
if(isclose(res1, res3)):
    print("Test - Matrix EL Norm P: PASSED")
else:
    print("Test - Matrix EL Norm P: NOT PASSED")

#------TEST - Matrix Mul Scalar----------------------------------------------
res1 = (m1_FL * 2.0)
res2 = (m1_CL * 2.0)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix CL Mul Scalar: PASSED")
else:
    print("Test - Matrix CL Mul Scalar: NOT PASSED")   

res1 = (m1_FL * 2.0)
res2 = (m1_EL * 2.0)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix EL Mul Scalar: PASSED")
else:
    print("Test - Matrix EL Mul Scalar: NOT PASSED")     

#------TEST - Matrix Add Scalar----------------------------------------------
res1 = (m1_FL + 2.0)
res2 = (m1_CL + 2.0)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix CL Add Scalar: PASSED")
else:
    print("Test - Matrix CL Add Scalar: NOT PASSED")   

res1 = (m1_FL + 2.0)
res2 = (m1_EL + 2.0)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix EL Add Scalar: PASSED")
else:
    print("Test - Matrix EL Add Scalar: NOT PASSED") 
    
#------TEST - Matrix Sub Scalar----------------------------------------------
res1 = (m1_FL - 2.0)
res2 = (m1_CL - 2.0)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix CL Sub Scalar: PASSED")
else:
    print("Test - Matrix CL Sub Scalar: NOT PASSED")   

res1 = (m1_FL - 2.0)
res2 = (m1_EL - 2.0)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix EL Sub Scalar: PASSED")
else:
    print("Test - Matrix EL Sub Scalar: NOT PASSED")     

#------TEST - Matrix Add Matrix----------------------------------------------
res1 = (m1_FL + m2_FL)
res2 = (m1_CL + m2_CL)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        print(res1)
        m1_CL.Dump()
        m1_EL.Dump()        
        m2_CL.Dump()        
        m2_EL.Dump()
        quit()
        break
    
if(not flag):
    print("Test - Matrix CL Add Matrix: PASSED")
else:
    print("Test - Matrix CL Add Matrix: NOT PASSED") 

res1 = (m1_FL + m2_FL)
res2 = (m1_EL + m2_EL)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        print(i, j)
        break
    
if(not flag):
    print("Test - Matrix EL Add Matrix: PASSED")
else:
    print("Test - Matrix EL Add Matrix: NOT PASSED") 

#------TEST - Matrix Sub Matrix----------------------------------------------
res1 = (m1_FL - m2_FL)
res2 = (m1_CL - m2_CL)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        print(res1)
        res2.Dump()
        res3 = (m1_EL - m2_EL)
        res3.Dump()         
        break
    
if(not flag):
    print("Test - Matrix CL Sub Matrix: PASSED")
else:
    print("Test - Matrix CL Sub Matrix: NOT PASSED") 

res1 = (m1_FL - m2_FL)
res2 = (m1_EL - m2_EL)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix EL Sub Matrix: PASSED")
else:
    print("Test - Matrix EL Sub Matrix: NOT PASSED") 
    
#------TEST - Matrix Swap Row / Col--------------------------------------------------

res1 = eilig.Matrix(m1_FL)
res2 = eilig.EllpackCL(kernels, m1_FL)
flag = False

for i in range (0, rows):
    m1 = random.randint(0, rows - 1)
    m2 = random.randint(0, rows - 1)
    n1 = random.randint(0, cols - 1)
    n2 = random.randint(0, cols - 1)

    res1.SwapRows(m1, m2)
    res1.SwapCols(n1, n2)

    res2.SwapRows(m1, m2)
    res2.SwapCols(n1, n2)
 
for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix CL Swap: PASSED")
else:
    print("Test - Matrix CL Swap: NOT PASSED") 
    
res1 = eilig.Matrix(m1_FL)
res2 = eilig.Ellpack(m1_FL)
flag = False

for i in range (0, rows):
    m1 = random.randint(0, rows - 1)
    m2 = random.randint(0, rows - 1)
    n1 = random.randint(0, cols - 1)
    n2 = random.randint(0, cols - 1)

    res1.SwapRows(m1, m2)
    res1.SwapCols(n1, n2)

    res2.SwapRows(m1, m2)
    res2.SwapCols(n1, n2)
 
for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix EL Swap: PASSED")
else:
    print("Test - Matrix EL Swap: NOT PASSED")  
 
#------TEST  - Matrix Fill-----------------------------------------------------
res1 = eilig.Matrix(rows, cols)
res2 = eilig.EllpackCL(kernels, rows, cols)

res1.Fill(2.0)
res2.Fill(2.0)

flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i, j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix CL Fill: PASSED")
else:
    print("Test - Matrix CL Fill: NOT PASSED")
    
res1 = eilig.Matrix(rows, cols)
res2 = eilig.Ellpack(rows, cols)

res1.Fill(2.0)
res2.Fill(2.0)
flag = False

for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i, j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break    
        
if(not flag):
    print("Test - Matrix EL Fill: PASSED")
else:
    print("Test - Matrix EL Fill: NOT PASSED")

#------TEST - Matrix Transpose-------------------------------------------------
res1 = m1_FL.Transpose()
res2 = m1_CL.Transpose()
res3 = m1_EL.Transpose()

passed = True
for i in range (0, cols): 
    for j in range (0, rows):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            passed = False
            break
    
    if(passed == False):
        break
    
if(passed):
    print("Test - Matrix CL Transpose: PASSED")
else:
    print("Test - Matrix CL Transpose: NOT PASSED")

passed = True
for i in range (0, cols): 
    for j in range (0, rows):  
        if(res1.GetValue(i,j) != res3.GetValue(i, j)):
            passed = False
            break
    
    if(passed == False):
        break
    
if(passed):
    print("Test - Matrix EL Transpose: PASSED")
else:
    print("Test - Matrix EL Transpose: NOT PASSED")

#------TEST - Matrix Vector Multiplication-------------------------------------
res1 = m1_FL * v2_FL
res2 = m1_CL * v2_CL
res3 = m1_EL * v2_EL

passed = True
for i in range (0, rows): 
    if(not isclose(res1.GetValue(i), res2.GetValue(i))):
        passed = False
        break
    
if(passed):
    print("Test - Matrix CL Vector Multiplication: PASSED")
else:
    print("Test - Matrix CL Vector Multiplication: NOT PASSED")
    

passed = True
for i in range (0, rows): 
    if(not isclose(res1.GetValue(i), res3.GetValue(i))):
        passed = False
        break
    
if(passed):
    print("Test - Matrix EL Vector Multiplication: PASSED")
else:
    print("Test - Matrix EL Vector Multiplication: NOT PASSED")        

#------TEST - Matrix Diagonal---------------------------------------------------
res1 = m1_FL.Diagonal()
res2 = m1_CL.Diagonal()
res3 = m1_EL.Diagonal()

flag = False
for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res2.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix CL Diagonal: PASSED")
else:
    print("Test - Matrix CL Diagonal: NOT PASSED")

flag = False
for i in range (0, rows): 
    flag = False
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res3.GetValue(i, j)):
            flag = True
            break
    
    if(flag == True):
        break
    
if(not flag):
    print("Test - Matrix EL Diagonal: PASSED")
else:
    print("Test - Matrix EL Diagonal: NOT PASSED")

#------TEST - Matrix Lower------------------------------------------------------
res1 = m1_FL.Lower(False)
res2 = m1_CL.Lower(False)
res3 = m1_EL.Lower(False)

passed = True
for i in range (0, rows): 
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res3.GetValue(i, j)):
            passed = False
            break
   
    if(passed == False):
        break
    
if(passed):
    print("Test - Matrix CL Lower: PASSED")
else:
    print("Test - Matrix CL Lower: NOT PASSED")

passed = True
for i in range (0, rows): 
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res3.GetValue(i, j)):
            passed = False
            break
    
    if(passed == False):
        break
  
if(passed):
    print("Test - Matrix EL Lower: PASSED")
else:
    print("Test - Matrix EL Lower: NOT PASSED")

#------TEST - Matrix Upper-----------------------------------------------------
res1 = m1_FL.Upper(False)
res2 = m1_CL.Upper(False)
res3 = m1_EL.Upper(False)

passed = True
for i in range (0, rows): 
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res3.GetValue(i, j)):
            passed = False
            break
   
    if(passed == False):
        break
    
if(passed):
    print("Test - Matrix CL Upper: PASSED")
else:
    print("Test - Matrix CL Upper: NOT PASSED")

passed = True
for i in range (0, rows): 
    for j in range (0, cols):  
        if(res1.GetValue(i,j) != res3.GetValue(i, j)):
            passed = False
            break
    
    if(passed == False):
        break
  
if(passed):
    print("Test - Matrix EL Upper: PASSED")
else:
    print("Test - Matrix EL Upper: NOT PASSED")
