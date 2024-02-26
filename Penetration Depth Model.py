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




#### Set number of model steps
nsteps = 20000 ;

#### Set timestep size (dt, in years)

#### Set depth of each layer (100m)
dz = 20
# Depths of each layer
#top 200m profile
#depths = np.arange (0, 220, dz)

#full basin profile
depths200m = np.array([10,30,50,70,90,110,130,150,170,190])
depths = np.array([10,30,50,70,90,110,130,150,170,190,700,1700])  # Adjusted for all layers (m)
volumes = 1e6 * np.array([8e3,8e3,8e3,8e3,8e3,8e3,8e3,8e3,8e3,8e3,20e3,20e3])  # in m3

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
Oxygen11 = np.empty(nsteps)
Oxygen12 = np.empty(nsteps)
### oxygen concs
OxygenConc1 = np.empty(nsteps) 
OxygenConc2 = np.empty(nsteps)
OxygenConc3 = np.empty(nsteps)
OxygenConc4 = np.empty(nsteps)
OxygenConc5 = np.empty(nsteps)
OxygenConc6 = np.empty(nsteps)
OxygenConc7 = np.empty(nsteps)
OxygenConc8 = np.empty(nsteps)
OxygenConc9 = np.empty(nsteps)
OxygenConc10 = np.empty(nsteps)
OxygenConc11 = np.empty(nsteps)
OxygenConc12 = np.empty(nsteps)

#Mean oxygen data 1955-2021
#units: days since 1950-01-01

####HAS FILLED IN GAPS WITH -9999

#Oxygencsv = Dataset("blksea_omi_health_oxygen_trend_annual_P20220610.nc")
#print(Oxygencsv)
#print(Oxygencsv['time'])

# Get the two variables you want to convert to NumPy arrays
#oxygen_inventory_mean = Oxygencsv.variables['oxygen_inventory_mean']
#time = Oxygencsv.variables['time']

# Convert the variables to NumPy arrays
#array1 = oxygen_inventory_mean[:]
#array2 = time[:]

# Create a new NumPy array with the two variables
#OxygenConc = np.stack((array1, array2/365 + 1950), axis=-1)


# Replace -9999 with a blank string
#OxygenConc = np.where(OxygenConc == -9999, '', OxygenConc)

# Print the new array
#print(OxygenConc)


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

#TOTAL NITROGEN INPUT (DON + DIN) - TO BE USED IN MODEL AS IS THE LIMITING NUTRIENT IN RESPIRATION
inputs1970= (DIN_input_total[0] + DON_input_total[0])* 3.6e7 #mol
inputs2000= (DIN_input_total[1] + DON_input_total[1])* 3.6e7 #mol
inputs2030= (DIN_input_total[2]+ DON_input_total[2])* 3.6e7 #mol
inputs2050= (DIN_input_total[3]+ DON_input_total[3])* 3.6e7 #mol
total_nutrient_input_kT = np.array([ inputs1970/3.6e7  , inputs2000/3.6e7  , inputs2030/3.6e7 , inputs2050/3.6e7  ])
total_nutrient_input = total_nutrient_input_kT * 3.6e7 #mol
nutrient_input_timeline = np.column_stack((input_times, total_nutrient_input)) #mol / year


# Create a Nitrogen figure and axis
fig, ax = plt.subplots()

# Plot the lines
ax.plot(input_times, total_nutrient_input, label='Total Nitrogen Input')
ax.plot(input_times, DIN_input_total*3.6e7, label=' Total DIN Input')
ax.plot(input_times, DON_input_total*3.6e7, label='Total DON Input')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xlabel('Years')
ax.set_ylabel('Input (mol)')
ax.set_title('Nitrogen Inputs from 1970-2050')

#TOTAL PHOSPHORUS INPUT (DOP + DIP) 
Pinputs1970= (DIP_input_total[0] + DOP_input_total[0])* 3.6e7 #mol
Pinputs2000= (DIP_input_total[1] + DOP_input_total[1])* 3.6e7 #mol
Pinputs2030= (DIP_input_total[2]+ DOP_input_total[2])* 3.6e7 #mol
Pinputs2050= (DIP_input_total[3]+ DOP_input_total[3])* 3.6e7 #mol
total_P_input_kT = np.array([ Pinputs1970/3.6e7  , Pinputs2000/3.6e7  , Pinputs2030/3.6e7 , Pinputs2050/3.6e7  ])
total_P_input = total_P_input_kT * 3.6e7 #mol


#Create a Phosphorus figure & axis
fig, ax = plt.subplots()
# Plot the lines
ax.plot(input_times, total_P_input, label='Total Phosphorus Input')
ax.plot(input_times, DIP_input_total*3.6e7, label=' Total DIP Input')
ax.plot(input_times, DOP_input_total*3.6e7, label='Total DOP Input')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xlabel('Years')
ax.set_ylabel('Input (mol)')
ax.set_title('Phosphorus Inputs from 1970-2050')

#SST data 1993-2022
#SeaSurfaceTemp= Dataset("Temperature over Time/SST 1993-2022")
#print(SeaSurfaceTemp)

#### Set timestep size (dt, in years) 1950-2050
t = np.empty(nsteps)
dt = 1e-5 ;
t[0] = 1950
# Model of oxygen penetration depth over time

#maxoxygendepth = np.empty(nsteps)

### set initial oxygen mol
#Oxygen conc: mol/m3
#Oxygen1 = 0-20m
Oxygen1[0] = 2.58828E+9 ; #mol
OxygenConc1[0] = Oxygen1[0] / volumes[0] ;
#Oxygen2 = 20-40m
Oxygen2[0] = 2.5114E+9 ;
OxygenConc2[0] = Oxygen2[0] / volumes[0] ;
#Oxygen3 = 40-60m
Oxygen3[0] = 2.29339E+9 ;
OxygenConc3[0] = Oxygen3[0] / volumes[0] ;
#Oxygen4 = 60-80m
Oxygen4[0] = 1.98118E+9 ;
OxygenConc4[0] = Oxygen4[0] / volumes[0] ;
#Oxygen5 = 80-100m
Oxygen5[0] = 7.43563E+8 ;
OxygenConc5[0] = Oxygen5[0] / volumes[0] ;
#Oxygen6 = 100-120m
Oxygen6[0] = 1.64769E+8 ;
OxygenConc6[0] = Oxygen6[0] / volumes[0] ;
#Oxygen7 = 120-140m
Oxygen7[0] = 5.65211E+7 ;
OxygenConc7[0] = Oxygen7[0] / volumes[0] ;
#Oxygen8 = 140-160m
Oxygen8[0] = 4.23908E+7 ;
OxygenConc8[0] = Oxygen8[0] / volumes[0] ;
#Oxygen9 = 160-180m
Oxygen9[0] = 1e4 ;
OxygenConc9[0] = Oxygen9[0] / volumes[0] ;
#Oxygen10 = 180-200m
Oxygen10[0] = 1e4 ;
OxygenConc10[0] = Oxygen10[0] / volumes[0] ;
#Oxygen11 = 200-1200m
Oxygen11[0] = 1e4;
OxygenConc11[0] = Oxygen11[0] / volumes[0] ;
#Oxygen12 = 1200-2200m
Oxygen12[0] = 1e4 ;
OxygenConc12[0] = Oxygen12[0] / volumes[0] ;

    # Update the maximum oxygen depth
#penetration_depths = []


for n in np.arange(0, nsteps, 1):
    # Calculate oxygen model parameters and processes at this timestep AND DEPTH OF EACH LAYER (dz)
    ##NEED EQUATION FOR RESPIRATION
    
    
    OxygenConc1[n]=( Oxygen1[n]/volumes[0] ) #mol/m3
    OxygenConc2[n]=( Oxygen2[n]/volumes[1] )
    OxygenConc3[n]=( Oxygen3[n]/volumes[2] )
    OxygenConc4[n]=( Oxygen4[n]/volumes[3] )
    OxygenConc5[n]=( Oxygen5[n]/volumes[4] )
    OxygenConc6[n]=( Oxygen6[n]/volumes[5] )
    OxygenConc7[n]=( Oxygen7[n]/volumes[6] )
    OxygenConc8[n]=( Oxygen8[n]/volumes[7] )
    OxygenConc9[n]=( Oxygen9[n]/volumes[8] )
    OxygenConc10[n]=( Oxygen10[n]/volumes[9] )
    OxygenConc11[n]=( Oxygen11[n]/volumes[10] )
    OxygenConc12[n]=( Oxygen12[n]/volumes[11] )

    
  #  mols of oxygen moved per year
# box 1 =  0.037487015
#box 2 = 0.014982349
#box 3 = 0.008667563
#box 4 = 0.006129205
#box 5 = 0.008853297
#box 6 = 0.012190083
#box 7 = 0.023961459
#box 8 = 0.151842974
#box 9 = 0.532798047
#box 10 = 2.537720471
#box 11 = 0
#box 12 = 0

#m3 of water moved via advection per year
# box 1 = 6.25E+04
# box 2 = 4.41E+04
# box 3 = 3.10E+04
# box 4 = 1.86E+04
# box 5 = 1.70E+04
# box 6 = 2.24E+04
# box 7 = 3.00E+04
# box 8 = 3.91E+04
# box 9 = 4.82E+04
# box 10 = 5.73E+04
# box 11= 9.72E+06
# box 12= 6.42E+06

    #### Calculate Oxygen model parameters and processes at this timestep = water flux (m3) * concentraiton (mol/m3)
    #Airsea = 4.72345E+12 ; 
    circ = 1;
    
    Airsea = 4.3E+11
    Down_1 = 6.25E+12 * OxygenConc1[n] * circ ; 
    Down_2 = 4.41E+12 * OxygenConc2[n] * circ;
    Down_3 = 3.10E+12 * OxygenConc3[n] * circ;
    Down_4 = 1.86E+12 * OxygenConc4[n] * circ;
    Down_5 = 1.70E+12 * OxygenConc5[n] * circ;
    Down_6 = 2.24E+12 * OxygenConc6[n] * circ;
    Down_7 = 3.00E+12 * OxygenConc7[n] * circ;
    Down_8 = 3.91E+12 * OxygenConc8[n] * circ;
    Down_9 = 4.82E+12 * OxygenConc9[n] * circ;
    Down_10 = 5.73E+12 * OxygenConc10[n] * circ;
    Down_11 = 9.72E+14 * OxygenConc11[n] * circ;
   

    Up_2 = 4.41E+12 * OxygenConc2[n] * circ;
    Up_3 = 3.10E+12 * OxygenConc3[n] * circ;
    Up_4 = 1.86E+12 * OxygenConc4[n] * circ;
    Up_5 = 1.70E+12 * OxygenConc5[n] * circ;
    Up_6 = 2.24E+12 * OxygenConc6[n] * circ;
    Up_7 = 3.00E+12 * OxygenConc7[n] * circ;
    Up_8 = 3.91E+12 * OxygenConc8[n] * circ;
    Up_9 = 4.82E+12 * OxygenConc9[n] * circ;
    Up_10 = 5.73E+12 * OxygenConc10[n] * circ;
    Up_11 = 9.72E+14 * OxygenConc11[n] * circ;
    Up_12 = 6.42E+14 * OxygenConc12[n] * circ;
    
    #Down_1 = 1 * ( Oxygen1[n]/Oxygen1[0] ) ; 
    #Down_2 = 1 * ( Oxygen2[n]/Oxygen2[0] ) ;
    #Down_3 = 1 * ( Oxygen3[n]/Oxygen3[0] ) ;
    #own_4 = 1 * ( Oxygen4[n]/Oxygen4[0] ) ;
    #Down_5 = 1 * ( Oxygen5[n]/Oxygen5[0] ) ;
    #Down_6 = 1 * ( Oxygen6[n]/Oxygen6[0] ) ;
    #Down_7 = 1 * ( Oxygen7[n]/Oxygen7[0] ) ;
    #Down_8 = 1 * ( Oxygen8[n]/Oxygen8[0] ) ;
    #Down_9 = 1 * ( Oxygen9[n]/Oxygen9[0] ) ;
    #Down_9 = 1 * ( Oxygen9[n]/Oxygen9[0] ) ;
    #Down_10 = 1 * ( Oxygen10[n]/Oxygen10[0] ) ;
    #Down_11 = 1 * ( Oxygen11[n]/Oxygen11[0] ) ;

    #Up_2 = 1 * ( Oxygen2[n]/Oxygen2[0] ) ;
    #Up_3 = 1 * ( Oxygen3[n]/Oxygen3[0] ) ;
    #Up_4 = 1 * ( Oxygen4[n]/Oxygen4[0] ) ;
    #Up_5 = 1 * ( Oxygen5[n]/Oxygen5[0] ) ;
    #Up_6 = 1 * ( Oxygen6[n]/Oxygen6[0] ) ;
    #Up_7 = 1 * ( Oxygen7[n]/Oxygen7[0] ) ;
    #Up_8 = 1 * ( Oxygen8[n]/Oxygen8[0] ) ;
    #Up_9 = 1 * ( Oxygen9[n]/Oxygen9[0] ) ;
    #Up_10 = 1 * ( Oxygen10[n]/Oxygen10[0] ) ;
    #Up_11 = 1 * ( Oxygen11[n]/Oxygen11[0] ) ;
    #Up_12 = 1 * ( Oxygen12[n]/Oxygen12[0] ) ;


    # Update the model reservoirs by adding and subtracting sources and sinks
    #### We multiply by dt because each source or sink process is defined in Gt of carbon per year
    #### On the final model step (n = steps) we do not calculate the future reservoir sizes, hence the 'if' statement
    #does this not need to be multiplied by oxygen concentration to give the mols of oxygen leaving
    k_resp = 0.75 ;
    Resp_1 = k_resp * inputs2000 *106/16 * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc1[n]) ))))  ; # * nutrient input of each year (currently 2000)
    Resp_2 = k_resp * inputs2000 *106/16 * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc2[n]) ))));
    Resp_3 = k_resp * inputs2000 *106/16 * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc3[n]) ))));
    Resp_4 = k_resp * inputs2000 *106/16 * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc4[n]) ))));
    Resp_5 = k_resp * inputs2000 *106/16 * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc5[n]) ))));
    Resp_6 = k_resp * inputs2000 *106/16 * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc6[n]) ))));
    Resp_7 = k_resp * inputs2000 *106/16 * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc7[n]) ))));
    Resp_8 = k_resp * inputs2000 *106/16 * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc8[n]) ))));
    Resp_9 = k_resp * inputs2000 *106/16 * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc9[n]) ))));
    Resp_10 = k_resp * inputs2000 *106/16 * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc10[n]) ))));
    Resp_11 = k_resp * inputs2000 *106/16  * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc11[n]) ))));
    Resp_12 = k_resp* inputs2000  *106/16  * (1 / (1 + np.exp(-( 3 + np.log10(OxygenConc12[n]) ))));
    


    
    # Accounting for respiration
    # Oxygen1, Oxygen2, Oxygen3 < 0.00002



    # if n < nsteps-1:
    #     Oxygen1[n+1] = Oxygen1[n] + ( Airsea + Up_2 - Down_1 - Resp_1 ) * dt ;
    #     Oxygen2[n+1] = Oxygen2[n] + ( Down_1 - Down_2 - Up_2 + Up_3 - Resp_2 ) * dt ;
    #     Oxygen3[n+1] = Oxygen3[n] + ( Down_2 - Down_3 - Up_3 + Up_4 - Resp_3 ) * dt ;
    #     Oxygen4[n+1] = Oxygen4[n] + ( Down_3 - Down_4 - Up_4 + Up_5 - Resp_4 ) * dt
    #     Oxygen5[n+1] = Oxygen5[n] + ( Down_4 - Down_5 - Up_5 + Up_6 - Resp_5 ) * dt
    #     Oxygen6[n+1] = Oxygen6[n] + ( Down_5 - Down_6 - Up_6 + Up_7 - Resp_6) * dt
    #     Oxygen7[n+1] = Oxygen7[n] + ( Down_6 - Down_7 - Up_7 + Up_8 - Resp_7) * dt
    #     Oxygen8[n+1] = Oxygen8[n] + ( Down_7 - Down_8 - Up_8 + Up_9 - Resp_8) * dt
    #     Oxygen9[n+1] = Oxygen9[n] + ( Down_8 - Down_9 - Up_9 + Up_10 - Resp_9) * dt
    #     Oxygen10[n+1] = Oxygen10[n] + ( Down_9 - Up_10 + Up_11 - Down_10 - Resp_10) * dt
    #     Oxygen11[n+1] = Oxygen11[n] + ( Down_10 - Up_11 + Up_12 - Down_11 - Resp_11) * dt
    #     Oxygen12[n+1] = Oxygen12[n] + ( Down_11 - Up_12 - Resp_12) * dt
        
 
    resp_constant = 0.0001 ;
    if n < nsteps-1:
        if Oxygen1[n] >= resp_constant:
            Oxygen1[n+1] = Oxygen1[n] + ( Airsea + Up_2 - Down_1 - Resp_1 ) * dt ;
        else:
            Oxygen1[n+1] = Oxygen1[n] + ( Airsea + Up_2 - Down_1 ) * dt ;

        if Oxygen2[n] >= resp_constant:
            Oxygen2[n+1] = Oxygen2[n] + ( Down_1 - Down_2 - Up_2 + Up_3 - Resp_2 ) * dt ;
        else:
            Oxygen2[n+1] = Oxygen2[n] + ( Down_1 - Down_2 - Up_2 + Up_3 ) * dt ;
                
        if Oxygen3[n] >= resp_constant:
            Oxygen3[n+1] = Oxygen3[n] + ( Down_2 - Down_3 - Up_3 + Up_4 - Resp_3 ) * dt ;
        else:
            Oxygen3[n+1] = Oxygen3[n] + ( Down_2 - Down_3 - Up_3 + Up_4 ) * dt ;

        if Oxygen4[n] >= resp_constant:
            Oxygen4[n+1] = Oxygen4[n] + ( Down_3 - Down_4 - Up_4 + Up_5 - Resp_4 ) * dt
        else:
            Oxygen4[n+1] = Oxygen4[n] + ( Down_3 - Down_4 - Up_4 + Up_5 ) * dt

        if Oxygen5[n] >= resp_constant:
            Oxygen5[n+1] = Oxygen5[n] + ( Down_4 - Down_5 - Up_5 + Up_6 - Resp_5 ) * dt
        else:
            Oxygen5[n+1] = Oxygen5[n] + ( Down_4 - Down_5 - Up_5 + Up_6 ) * dt

        if Oxygen6[n] >= resp_constant:
            Oxygen6[n+1] = Oxygen6[n] + ( Down_5 - Down_6 - Up_6 + Up_7 - Resp_6) * dt
        else:
            Oxygen6[n+1] = Oxygen6[n] + ( Down_5 - Down_6 - Up_6 + Up_7 ) * dt

        if Oxygen7[n] >= resp_constant:
            Oxygen7[n+1] = Oxygen7[n] + ( Down_6 - Down_7 - Up_7 + Up_8 - Resp_7) * dt
        else:
            Oxygen7[n+1] = Oxygen7[n] + ( Down_6 - Down_7 - Up_7 + Up_8 ) * dt   
        
        if Oxygen8[n] >= resp_constant:
            Oxygen8[n+1] = Oxygen8[n] + ( Down_7 - Down_8 - Up_8 + Up_9 - Resp_8) * dt
        else:
            Oxygen8[n+1] = Oxygen8[n] + ( Down_7 - Down_8 - Up_8 + Up_9) * dt
        
        if Oxygen9[n] >= resp_constant:
            Oxygen9[n+1] = Oxygen9[n] + ( Down_8 - Down_9 - Up_9 + Up_10 - Resp_9) * dt
        else:
            Oxygen9[n+1] = Oxygen9[n] + ( Down_8 - Down_9 - Up_9 + Up_10) * dt
        
        if Oxygen10[n] >= resp_constant:
            Oxygen10[n+1] = Oxygen10[n] + ( Down_9 - Up_10 + Up_11 - Down_10 - Resp_10) * dt
        else:
            Oxygen10[n+1] = Oxygen10[n] + ( Down_9 - Up_10 + Up_11 - Down_10) * dt
        
        if Oxygen11[n] >= resp_constant:
            Oxygen11[n+1] = Oxygen11[n] + ( Down_10 - Up_11 + Up_12 - Down_11 - Resp_11) * dt
        else:
            Oxygen11[n+1] = Oxygen11[n] + ( Down_10 - Up_11 + Up_12 - Down_11) * dt
        
        if Oxygen12[n] >= resp_constant:
            Oxygen12[n+1] = Oxygen12[n] + ( Down_11 - Up_12 - Resp_12) * dt
        else:
            Oxygen12[n+1] = Oxygen12[n] + ( Down_11 - Up_12 ) * dt
            
        #### Update model time         
        t[n+1] = t[n] + dt ;
        
        
        #### output time every 1000 steps
        if n % 1000 == 0:
            print(t[n])
        
        # #### speed up model
        # if t[n] > 1950.2:
        #     dt = 5e-4
                       
        
#### make a single large figure
fig = plt.figure(figsize=(25,20))


#### add first subplot in a 2x3 grid
plt.subplot(6,2,1)
plt.plot(t,Oxygen1)
plt.tick_params(axis='x', labelleft=False)
plt.ylabel('Oxygen1')

plt.subplot(6,2,2)
plt.plot(t,Oxygen2)
plt.ylabel('Oxygen2')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(6,2,3)
plt.plot(t,Oxygen3)
plt.ylabel('Oxygen3')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(6,2,4)
plt.plot(t,Oxygen4)
plt.ylabel('Oxygen4')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(6,2,5)
plt.plot(t,Oxygen5)
plt.ylabel('Oxygen5')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(6,2,6)
plt.plot(t,Oxygen6)
plt.ylabel('Oxygen6')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(6,2,7)
plt.plot(t,Oxygen7)
plt.ylabel('Oxygen7')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(6,2,8)
plt.plot(t,Oxygen8)
plt.ylabel('Oxygen8')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(6,2,9)
plt.plot(t,Oxygen9)
plt.ylabel('Oxygen9')
plt.tick_params(axis='x', labelleft=False)

plt.subplot(6,2,10)
plt.plot(t,Oxygen10)
plt.ylabel('Oxygen10')
plt.xlabel('Year')

plt.subplot(6,2, 11)
plt.plot(t,Oxygen11)
plt.ylabel('Oxygen11')
plt.xlabel('Year')

plt.subplot(6,2, 12)
plt.plot(t,Oxygen12)
plt.ylabel('Oxygen12')
plt.xlabel('Year')

# Adjust the spacing between the subplots
plt.subplots_adjust(wspace=0.5, hspace=1)
fig.tight_layout()


oxygen_depth_profile = np.array([OxygenConc1[nsteps-1], OxygenConc2[nsteps-1], OxygenConc3[nsteps-1], OxygenConc4[nsteps-1], OxygenConc5[nsteps-1], OxygenConc6[nsteps-1],
                                 OxygenConc7[nsteps-1], OxygenConc8[nsteps-1], OxygenConc9[nsteps-1], OxygenConc10[nsteps-1], OxygenConc11[nsteps-1], OxygenConc12[nsteps-1]])
oxygen_depth_profile_200m = np.array([OxygenConc1[nsteps-1], OxygenConc2[nsteps-1], OxygenConc3[nsteps-1], OxygenConc4[nsteps-1], OxygenConc5[nsteps-1], OxygenConc6[nsteps-1],
                                 OxygenConc7[nsteps-1], OxygenConc8[nsteps-1], OxygenConc9[nsteps-1], OxygenConc10[nsteps-1]])

plt.figure(figsize=(8, 6))
plt.plot(oxygen_depth_profile_200m, depths200m, marker='o', linestyle='-')
plt.title('Oxygen Depth Profile')
plt.xlabel('Oxygen Concentration (mol/m3)')
plt.ylabel('Depth (m)')
plt.grid(True)
plt.gca().invert_yaxis()  # Invert y-axis to represent deeper depths at the bottom
plt.show()

  
threshold = 0.00002

# Find the index where oxygen concentration drops below the threshold
index_below_threshold = np.argmax(oxygen_depth_profile < threshold)

# Depth where oxygen concentration first drops below the threshold
oxygen_penetration_depth = depths[index_below_threshold]

print(f"The depth at which oxygen concentration drops below {threshold} is approximately {oxygen_penetration_depth} meters.")
        

# Calculate the indices where Oxygen[i] < resp_constant for each time step
# indices = np.where(np.array([Oxygen1, Oxygen2, Oxygen3, Oxygen4, Oxygen5, Oxygen6, Oxygen7, Oxygen8, Oxygen9, Oxygen10, Oxygen11, Oxygen12]) < resp_constant)[1:] + np.arange(nsteps)[:, None]

# Calculate the depths where oxygen drops below the respconstant
# depths_below_resp_constant = indices * 20

# Find the first depths where oxygen drops below the resp_constant
# first_depths_below_resp_constant = depths_below_resp_constant[np.argsort(indices[:, 0])[0]]

# print("First Depths Where Oxygen Drops Below the Respiration Constant: \n", first_depths_below_resp_constant)
# Convert the list to a NumPy array
#penetration_depths = np.array(penetration_depths)
    
#plt.plot(t[:-1], penetration_depths[:-1])
#plt.xlabel('Time (years)')
#plt.ylabel('Oxygen Penetration Depth (m)')
#plt.title('Oxygen Penetration Depth Over Time')
#plt.grid()
#plt.show()
# ma