#!/usr/bin/env python

#Example for turbine distributor pressure field with 24x wicket gates
#fileIn contains of a pressure field sector (15°)
#------------------------------------------------------------------------------

import eilig
import math

fileIn  = 'fem.BANDWG'
fileOut = 'fem.BANDWG.out'
status  = 0
 
g       = 9.8081    # Gravity
rho     = 999.4     # Water density
h_ATM   = 10.08     # Atmosphere pressure
h_VAP   = 0.15      # water vapor pressure
KMLa    = 208.4     # Runner centerline elevation
TWL     = 214.0     # Tail water elevation

p_TWL   = rho * g * (TWL - KMLa)
p_ATM   = rho * g * h_ATM
p_VAP   = rho * g * h_VAP

data    = eilig.Matrix()
status  = eilig.ReadFromFile(data, fileIn)

data    = eilig.Add(data, p_TWL )                   #Add tail water pressure
data    = eilig.Add(data, p_ATM )                   #Convert to absolute pressure
data    = eilig.ClipSmallerThan(data, p_VAP, 3)     #Clip pressure below vapor pressure
data    = eilig.Add(data, -p_ATM )                  #Convert to relativ pressure

output  = eilig.Matrix(data)

z = 24
step = 360.0/z
angles = [i * step for i in range(1, z, 1)] 

for angle in angles:  
    dataRotated = eilig.RotatePoints(data, eilig.axis_z, math.radians(angle))
    output = eilig.Append(output, dataRotated)

eilig.WriteToFile(output, fileOut)