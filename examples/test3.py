import math
import random
import eilig
import time

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return math.fabs(a-b) <= max(rel_tol * max(math.fabs(a), math.fabs(b)), abs_tol)

platformNumber = 0
deviceNumber = 0

kernels = eilig.CreateKernels("kernels.c", platformNumber, deviceNumber);
rows_max = 50
cols_max = 50

numberTests = 100
numberFills = 1000
passedCL = True
passedEL = True

for k in range(0, numberTests):
    
    rows = random.randint(1, rows_max)      
    cols = random.randint(1, cols_max)   
    
    m1_FL = eilig.Matrix(rows, cols)    
    m1_CL = eilig.EllpackCL(kernels, rows, cols)
    m1_EL = eilig.Ellpack(rows, cols)
    
    for n in range (0, random.randint(1, numberFills)): 
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)
        
        m1_FL.SetValue(i, j, 1.0)
        m1_CL.SetValue(i, j, 1.0)
        m1_EL.SetValue(i, j, 1.0)
    
    for i in range (0, rows): 
        for j in range (0, cols):  
            if(not isclose(m1_CL.GetValue(i, j), m1_FL.GetValue(i, j)) ):
                passedCL = False
                break
    
        if(passedCL == False):
            break;

    if(passedCL):
        print("Test - Matrix CL Init: PASSED")
    else:
        print("Test - Matrix CL Init: NOT PASSED ")
        print(m1_FL)
        print(m1_CL)
        m1_CL.Dump()
        break
            
    for i in range (0, rows): 
        for j in range (0, cols):  
            if(not isclose(m1_EL.GetValue(i, j), m1_FL.GetValue(i, j)) ):
                passedEL = False
                break
    
        if(passedEL == False):
            break;            

    if(passedEL):
        print("Test - Matrix EL Init: PASSED")
    else:
        print("Test - Matrix EL Init: NOT PASSED ")
        m1_EL.Dump()
        break           
