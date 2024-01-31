#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:45:14 2024

@author: ellarea
"""
#### Import the modules that we need - only run this cell once
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import matplotlib.gridspec as gridspec



#### Set number of model steps
nsteps = 10000 ;

#### Set timestep size (dt, in years)
t = np.empty(nsteps)
dt = 0.1 ;
t[0] = 1950

#### Set depth of each layer (100m)
d = np.empty(nsteps)
dz = 100
d[0] = 0

#### Make an empty array to hold the model results 
Oxygen1 = np.empty(nsteps) 
Oxygen2 = np.empty(nsteps)
Oxygen3 = np.empty(nsteps)
Oxygen4 = np.empty(nsteps)
Oxygen5 = np.empty(nsteps)
Oxygen6 = np.empty(nsteps)
Oxygen7 = np.empty(nsteps)
Oxygen8 = np.empty(nsteps)
Oxygen9 = np.empty(nsteps)
Oxygen10 = np.empty(nsteps)

#Mean oxygen data 1955-2021
#units: days since 1950-01-01

####HAS FILLED IN GAPS WITH -9999

Oxygencsv = Dataset("blksea_omi_health_oxygen_trend_annual_P20220610.nc")
print(Oxygencsv)
print(Oxygencsv['time'])

# Get the two variables you want to convert to NumPy arrays
oxygen_inventory_mean = Oxygencsv.variables['oxygen_inventory_mean']
time = Oxygencsv.variables['time']

# Convert the variables to NumPy arrays
array1 = oxygen_inventory_mean[:]
array2 = time[:]

# Create a new NumPy array with the two variables
OxygenConc = np.stack((array1, array2/365 + 1950), axis=-1)


# Replace -9999 with a blank string
OxygenConc = np.where(OxygenConc == -9999, '', OxygenConc)

# Print the new array
print(OxygenConc)


#Nutrient inputs 1970 - 2050
DINInputsNorthcsv = pd.read_csv("DINNorthBlackSeaInputs.csv", header = None)
DINInputsNorth = DINInputsNorthcsv.values
print(DINInputsNorth)

DINInputsSouthcsv= pd.read_csv("DINSouthBlackSeaInputs.csv", header = None)
DINInputsSouth = DINInputsSouthcsv.values
print(DINInputsSouth)


DINInputsAzovcsv= pd.read_csv("DINAzovSeaInputs.csv", header = None)
DINInputsAzov = DINInputsAzovcsv.values
print(DINInputsAzov)


DONInputsNorthcsv= pd.read_csv("DONNorthBlackSeaInputs.csv", header = None)
DONInputsNorth = DONInputsNorthcsv.values
print(DONInputsNorth)


DONInputSouthcsv= pd.read_csv("DONSouthBlackSeaInputs.csv", header = None)
DONInputsSouth = DONInputSouthcsv.values
print(DONInputsSouth)


DONInputAzovcsv= pd.read_csv("DONAzovSeaInputs.csv", header = None)
DONInputsAzov = DONInputAzovcsv.values
print(DONInputsAzov)


DIPInputsNorthcsv= pd.read_csv("DIPNorthBlackSeaInputs.csv", header = None)
DIPInputsNorth = DIPInputsNorthcsv.values
print(DIPInputsNorth)


DIPInputsSouthcsv= pd.read_csv("DIPSouthBlackSeaInputs.csv", header = None)
DIPInputsSouth = DIPInputsSouthcsv.values
print(DIPInputsSouth)


DIPInputsAzovcsv= pd.read_csv("DIPAzovSeaInputs.csv", header = None)
DIPInputsAzov = DIPInputsAzovcsv.values
print(DIPInputsAzov)


DOPInputsNorthcsv= pd.read_csv("DOPNorthBlackSeaInputs.csv", header = None)
DOPInputsNorth = DOPInputsNorthcsv.values
print(DOPInputsNorth)



DOPInputsSouthcsv= pd.read_csv("DOPSouthBlackSeaInputs.csv", header = None)
DOPInputsSouth = DOPInputsSouthcsv.values
print(DOPInputsSouth)


DOPInputsAzovcsv= pd.read_csv("DOPAzovSeaInputs.csv", header = None)
DOPInputsAzov = DOPInputsAzovcsv.values
print(DOPInputsAzov)

#COMBINING ALL NUTRIENT AREA INPUTS

DOPNutrientInputs = DOPInputsAzov + DOPInputsNorth + DOPInputsSouth
# Convert the NumPy array back to a DataFrame
DOPNutrientInputs_df = pd.DataFrame(DOPNutrientInputs, columns=['0','1' ])

# Add a new column to the DataFrame
DOPNutrientInputs_df['NewColumn'] = np.array([1970, 2000, '2030AM', '2030GO', '2050AM', '2050GO'])  # Replace this with your desired values

# Convert the DataFrame back to a NumPy array
DOPNutrientInputs = DOPNutrientInputs_df.values

# Print the NumPy array
print(DOPNutrientInputs)

### do averages for 2030 and 2050, and add years
DOP_input_total = np.array([ DOPNutrientInputs[0,1] , DOPNutrientInputs[1,1] , ( DOPNutrientInputs[2,1] + DOPNutrientInputs[3,1] )/2 , (DOPNutrientInputs[4,1] + DOPNutrientInputs[5,1])/2 ])
input_times = np.array([1970 , 2000 , 2030, 2050])
DOPInputTimeline = np.column_stack((input_times, DOP_input_total))



DIPNutrientInputs = DIPInputsAzov + DIPInputsNorth + DIPInputsSouth
# Convert the NumPy array back to a DataFrame
DIPNutrientInputs_df = pd.DataFrame(DIPNutrientInputs, columns=['0','1' ])

# Add a new column to the DataFrame
DIPNutrientInputs_df['NewColumn'] = np.array([1970, 2000, '2030AM', '2030GO', '2050AM', '2050GO'])  # Replace this with your desired values

# Convert the DataFrame back to a NumPy array
DIPNutrientInputs = DIPNutrientInputs_df.values

# Print the NumPy array
print(DIPNutrientInputs)

### do averages for 2030 and 2050, and add years
DIP_input_total = np.array([ DIPNutrientInputs[0,1] , DIPNutrientInputs[1,1] , ( DIPNutrientInputs[2,1] + DIPNutrientInputs[3,1] )/2 , (DIPNutrientInputs[4,1] + DIPNutrientInputs[5,1])/2 ])
input_times = np.array([1970 , 2000 , 2030, 2050])
DIPInputTimeline = np.column_stack((input_times, DIP_input_total))



DONNutrientInputs = DONInputsAzov + DONInputsSouth + DONInputsNorth
# Convert the NumPy array back to a DataFrame
DONNutrientInputs_df = pd.DataFrame(DONNutrientInputs, columns=['0','1' ])

# Add a new column to the DataFrame
DONNutrientInputs_df['NewColumn'] = np.array([1970, 2000, '2030AM', '2030GO', '2050AM', '2050GO'])  # Replace this with your desired values

# Convert the DataFrame back to a NumPy array
DONNutrientInputs = DONNutrientInputs_df.values

# Print the NumPy array
print(DONNutrientInputs)

### do averages for 2030 and 2050, and add years
DON_input_total = np.array([ DONNutrientInputs[0,1] , DONNutrientInputs[1,1] , ( DONNutrientInputs[2,1] + DONNutrientInputs[3,1] )/2 , (DONNutrientInputs[4,1] + DONNutrientInputs[5,1])/2 ])
input_times = np.array([1970 , 2000 , 2030, 2050])
DONInputTimeline = np.column_stack((input_times, DON_input_total))



DINNutrientInputs = DINInputsAzov + DINInputsNorth + DINInputsSouth
# Convert the NumPy array back to a DataFrame
DINNutrientInputs_df = pd.DataFrame(DINNutrientInputs, columns=['0','1' ])

# Add a new column to the DataFrame
DINNutrientInputs_df['NewColumn'] = np.array([1970, 2000, '2030AM', '2030GO', '2050AM', '2050GO'])  # Replace this with your desired values

# Convert the DataFrame back to a NumPy array
DINNutrientInputs = DINNutrientInputs_df.values

# Print the NumPy array
print(DINNutrientInputs)


### do averages for 2030 and 2050, and add years
DIN_input_total = np.array([ DINNutrientInputs[0,1] , DINNutrientInputs[1,1] , ( DINNutrientInputs[2,1] + DINNutrientInputs[3,1] )/2 , (DINNutrientInputs[4,1] + DINNutrientInputs[5,1])/2 ])
input_times = np.array([1970 , 2000 , 2030, 2050])
DINInputTimeline = np.column_stack((input_times, DIN_input_total))

#SST data 1993-2022
#SeaSurfaceTemp= Dataset("Temperature over Time/SST 1993-2022")
#print(SeaSurfaceTemp)

t = np.empty(nsteps)
# Model of oxygen penetration depth over time

maxoxygendepth = np.empty(nsteps)


### set initial oxygen concentration
Oxygen1[0] = 100 ;
Oxygen2[0] = 100 ;
Oxygen3[0] = 100 ;
Oxygen4[0] = 100 ;
Oxygen5[0] = 100 ;
Oxygen6[0] = 100 ;
Oxygen7[0] = 100 ;
Oxygen8[0] = 100 ;
Oxygen9[0] = 100 ;
Oxygen10[0] = 100 ;

for n in np.arange(0, nsteps, 1):
    # Calculate oxygen model parameters and processes at this timestep AND DEPTH OF EACH LAYER (dz)
    ##NEED EQUATION FOR RESPIRATION
    
    

    #### Calculate Oxygen model parameters and processes at this timestep
    Airsea = 1 ;
    Down_1 = 1 * ( Oxygen1[n]/Oxygen1[0] ) ; 
    Down_2 = 1 * ( Oxygen2[n]/Oxygen2[0] ) ;
    Down_3 = 1 * ( Oxygen3[n]/Oxygen3[0] ) ;
    Down_4 = 1 * ( Oxygen4[n]/Oxygen4[0] ) ;
    Down_5 = 1 * ( Oxygen5[n]/Oxygen5[0] ) ;
    Down_6 = 1 * ( Oxygen6[n]/Oxygen6[0] ) ;
    Down_7 = 1 * ( Oxygen7[n]/Oxygen7[0] ) ;
    Down_8 = 1 * ( Oxygen8[n]/Oxygen8[0] ) ;
    Down_9 = 1 * ( Oxygen9[n]/Oxygen9[0] ) ;

    Up_2 = 1 * ( Oxygen2[n]/Oxygen2[0] ) ;
    Up_3 = 1 * ( Oxygen3[n]/Oxygen3[0] ) ;
    Up_4 = 1 * ( Oxygen4[n]/Oxygen4[0] ) ;
    Up_5 = 1 * ( Oxygen5[n]/Oxygen5[0] ) ;
    Up_6 = 1 * ( Oxygen6[n]/Oxygen6[0] ) ;
    Up_7 = 1 * ( Oxygen7[n]/Oxygen7[0] ) ;
    Up_8 = 1 * ( Oxygen8[n]/Oxygen8[0] ) ;
    Up_9 = 1 * ( Oxygen9[n]/Oxygen9[0] ) ;
    Up_10 = 1 * ( Oxygen10[n]/Oxygen10[0] ) ;


    # Update the model reservoirs by adding and subtracting sources and sinks
    #### We multiply by dt because each source or sink process is defined in Gt of carbon per year
    #### On the final model step (n = steps) we do not calculate the future reservoir sizes, hence the 'if' statement
    k_resp = 0.1 ;
    Resp_1 = k_resp ;
    Resp_2 = k_resp ;
    Resp_3 = k_resp ;
    Resp_4 = k_resp ;
    Resp_5 = k_resp ;
    Resp_6 = k_resp ;
    Resp_7 = k_resp ;
    Resp_8 = k_resp ;
    Resp_9 = k_resp ;
    Resp_10 = k_resp ;

    
    # Accounting for respiration
    # Oxygen1, Oxygen2, Oxygen3 < 0.00002



    if n < nsteps-1:
        Oxygen1[n+1] = Oxygen1[n] + ( Airsea + Up_2 - Down_1 - Resp_1 ) * dt ;
        Oxygen2[n+1] = Oxygen2[n] + ( Down_1 - Down_2 - Up_2 + Up_3 - Resp_2 ) * dt ;
        Oxygen3[n+1] = Oxygen3[n] + ( Down_2 - Down_3 - Up_3 + Up_4 - Resp_3 ) * dt ;
        Oxygen4[n+1] = Oxygen4[n] + ( Down_3 - Down_4 - Up_4 + Up_5 - Resp_4 ) * dt
        Oxygen5[n+1] = Oxygen5[n] + ( Down_4 - Down_5 - Up_5 + Up_6 - Resp_5 ) * dt
        Oxygen6[n+1] = Oxygen6[n] + ( Down_5 - Down_6 - Up_6 + Up_7 - Resp_6) * dt
        Oxygen7[n+1] = Oxygen7[n] + ( Down_6 - Down_7 - Up_7 + Up_8 - Resp_7) * dt
        Oxygen8[n+1] = Oxygen8[n] + ( Down_7 - Down_8 - Up_8 + Up_9 - Resp_8) * dt
        Oxygen9[n+1] = Oxygen9[n] + ( Down_8 - Down_9 - Up_9 + Up_10 - Resp_9) * dt
        Oxygen10[n+1] = Oxygen10[n] + ( Down_9 - Up_10 - Resp_10) * dt
                
        #### Update model time         
        t[n+1] = t[n] + dt ;
        
#### make a single large figure
fig = plt.figure(figsize=(25,20))


#### add first subplot in a 2x3 grid
plt.subplot(10,1,1)
plt.plot(t,Oxygen1)
plt.tick_params(axis='x', labelleft=False)
plt.ylabel('Oxygen1')

plt.subplot(10,1,2)
plt.plot(t,Oxygen2)
plt.ylabel('Oxygen2')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(10,1,3)
plt.plot(t,Oxygen3)
plt.ylabel('Oxygen3')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(10,1,4)
plt.plot(t,Oxygen4)
plt.ylabel('Oxygen4')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(10,1,5)
plt.plot(t,Oxygen5)
plt.ylabel('Oxygen5')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(10,1,6)
plt.plot(t,Oxygen6)
plt.ylabel('Oxygen6')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(10,1,7)
plt.plot(t,Oxygen7)
plt.ylabel('Oxygen7')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(10,1,8)
plt.plot(t,Oxygen8)
plt.ylabel('Oxygen8')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(10,1,9)
plt.plot(t,Oxygen9)
plt.ylabel('Oxygen9')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(10,1,10)
plt.plot(t,Oxygen10)
plt.ylabel('Oxygen10')
plt.xlabel('Year')

plt.subplots_adjust(wspace=0.5, hspace=1)
  

        

    # Update the maximum oxygen depth
# maxoxygendepth[n] = 0 
 #for i in range(10):
        #if OxygenConc[i][n] > 0.00002:  # threshold value
            #maxoxygendepth[n] = i * dz  # dz is the depth of each layer
            #break
        


#### add first subplot in a 2x3 grid
# plt.plot(t, maxoxygendepth)
# plt.xlabel('Year')
# plt.ylabel('Oxygen')        
        
        