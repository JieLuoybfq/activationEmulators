# Suppress warnings
import warnings
warnings.simplefilter('ignore')

import pyrcel as pm
import numpy as np
import pandas as pd

from smt.sampling_methods import LHS

import datetime
startDT = datetime.datetime.now()
print (str(startDT))

S0 = 0.00   # Supersaturation, 1-RH

# Log10N: same as RW16: [1,4]
# Log10ug: same as RW16: [-3,1]
# Sigma_g: we only need 2 values: 1.6 and 1.8
# Kappa: same as RW16: [0,1.2]
# Log10V: same as RW16: [-2,1]
# T: Use a colder lower bound: [-233.310]
# P: Changed to match T range: [200, 1050], unit = hPa
# ac: same as RW16: [0.1,1]
        
# The limits of the LHS sampling 
varLimits = np.array([[1.0,4.0], [-3.0,1.0], [0.0,1.0], [0.0,1.2], [-2.0,1.0], [248.0,310.0], 
                      [50000.0,105000.0], [0.1,1]])

# varLimits = np.array([[1.0,4.0], [-3.0,1.0], [0.0,1.0], [0.0,1.2], [-2.0,1.0], [310.0,314.0], 
#                       [50000.0,105000.0], [0.1,1]])

# The log limits
# varLimits = np.array([[10**1.0,10**4.0], [10**-3.0,10**1.0], [0.0,1.0], [0.0,1.2], [10**-2.0,10**1.0], [248.0,310.0], 
#                       [50000.0,105000.0], [0.1,1]])

# The kappa limits
varLimits = np.array([[1.0,4.0], [-3.0,1.0], [0.0,1.0], [-11.0,-1.0], [-2.0,1.0], [248.0,310.0],
                      [50000.0,105000.0], [0.1,1]])

# Do the LHS for 10,000 points
sampling = LHS(xlimits=varLimits)

num = 2000
varSamp = sampling(num)

# Adjust the binary Sigma_g variable to either 1.6 or 1.8
varSamp[varSamp[:,2] >= 0.5, 2] = 1.8
varSamp[varSamp[:,2] < 0.5, 2] = 1.6

# Log limit adjustments
# varSamp[:,0] = np.log10(varSamp[:,0])
# varSamp[:,1] = np.log10(varSamp[:,1])
# varSamp[:,4] = np.log10(varSamp[:,4])

# Kappa limit adjustments
varSamp[:,3] = 10**varSamp[:,3]


from pyrcel import binned_activation

smaxes, act_fracs = [], []
smaxes_arg, act_fracs_arg = [], []
smaxes_mbn, act_fracs_mbn = [], []

# [0,      1,       2,       3,     4,      5, 6, 7]
# [Log10N, Log10ug, Sigma_g, Kappa, Log10V, T, P, ac]

# Loop over all intitial conditions 
for Var in varSamp:
    # print(Var)
    try:
        # This throws an error: CVodeError: 'Convergence test failures...'
        # Var = np.array([2.531650e+00, 5.940000e-02, 1.600000e+00, 2.239800e-01, 9.845500e-01,
        #               246.0, 8.564825e+04, 2.714950e-01]
    
        # This throws an error: ParcelModelError: "Couldn't calculate initial aerosol population wet sizes."
        # Var = np.array([3.228550e+00, -2.987400e+00, 1.800000e+00, 2.346000e-02, 
        #                 -1.184150e+00, 2.647555e+02, 8.913525e+04, 9.878950e-01])

        # Initialize Aerosol distritbution    

        aer =  pm.AerosolSpecies('ammonium sulfate', 
                                 pm.Lognorm(mu=10**Var[1], sigma=Var[2], N=10.0**Var[0]), 
                                 kappa=Var[3], bins=250)
        initial_aerosols = [aer]
 
        # Initialize the model
        model = pm.ParcelModel(initial_aerosols, 10**Var[4], Var[5], S0, Var[6], accom=Var[7], console=False)

        par_out, aer_out = model.run(t_end=600., dt=1.0, solver='cvode', 
                                     output='dataframes', terminate=True)
        # Extract the supersaturation/activation details from the model
        # output
        S_max = par_out['S'].max()
        time_at_Smax = par_out['S'].argmax()
        wet_sizes_at_Smax = aer_out['ammonium sulfate'].iloc[time_at_Smax]
        wet_sizes_at_Smax = np.array(wet_sizes_at_Smax.tolist())

        #print(S_max)
        #print(Var[5])
        #print(wet_sizes_at_Smax)
        #print(aer)

        frac_eq, _, _, _ = binned_activation(S_max, Var[5], wet_sizes_at_Smax, aer)
    except:
        S_max = -999.0            
        frac_eq = -999.0

    # Save the output
    smaxes.append(S_max)
    act_fracs.append(frac_eq)

    # ARG and MBN Schemes
    smax_arg, _, afs_arg = pm.arg2000(10**Var[4], Var[5], Var[6], initial_aerosols, accom=Var[7])
    smax_mbn, _, afs_mbn = pm.mbn2014(10**Var[4], Var[5], Var[6], initial_aerosols, accom=Var[7])

    smaxes_arg.append(smax_arg)
    act_fracs_arg.append(afs_arg[0])
    smaxes_mbn.append(smax_mbn)
    act_fracs_mbn.append(afs_mbn[0])
    
print('DONE! Saving output:')

aerData = {'smax' : smaxes, 'actFrac' : act_fracs,
           'Log10N' : varSamp[:,0], 'Log10ug' : varSamp[:,1], 
           'Sigma_g' : varSamp[:,2], 'Kappa' : varSamp[:,3], 
           'Log10V' : varSamp[:,4], 'T' : varSamp[:,5],
           'P' : varSamp[:,6], 'ac' : varSamp[:,7],
           'smaxes_arg' : smaxes_arg, 'actFrac_arg' : act_fracs_arg,
           'smaxes_mbn' : smaxes_mbn, 'actFrac_mbn' : act_fracs_mbn}
df = pd.DataFrame(aerData, columns= ['smax', 'actFrac', 'Log10N', 'Log10ug', 'Sigma_g', 
                                     'Kappa', 'Log10V', 'T', 'P', 'ac', 'smaxes_arg', 
                                     'actFrac_arg', 'smaxes_mbn', 'actFrac_mbn'])
df.to_csv('parcelOutputTest250BinsLowKappa.csv',index=False)

print('We did it!')

endDT = datetime.datetime.now()
print (str(endDT))


