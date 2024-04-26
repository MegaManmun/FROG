
#%%
# DEFINE PARAMETERS: -------------------------------------------------------------------------------

Flip_up_down = False  # To flip spectrogram in wavelength direction
Constant_wavelength = True  # To interpolate frog trace on constant wavelength scale from constant frequency scale
Plot_external_frog_trace = False  # To plot external FROG trace, and not this one which we convert
Time_scale_resample = False

time_step = 4 # time interval between points in fs
central_wavelength = 515*10**(-9)
c = 299792458

#%%
# LOAD MEASUREMENT FILES --------------------------------------------------------------------------

import torch
import matplotlib.pyplot as plt
import numpy as np
import re # Used to check if a string contains the specified search pattern
from tkinter.filedialog import askopenfilename  # Module that creates open dialog and return the opened file object in read-only mode

# Select three files:
filenames = askopenfilename(title="Select FROG trace, Gate file and Frequency file", multiple = True)

# Load files based on their names:
for _fn in filenames:  # Iterate through our tuple, which contains path+filename
    if re.search("FROG", _fn):  # Searches in the string for the word and returns True if it finds it
        frog_trace_filename = _fn  # Save file path+name
        frog_trace = np.genfromtxt(frog_trace_filename).astype('float32')  # Open the file as float 32
        frog_trace = np.flipud(frog_trace)
    if re.search("Gate", _fn):
        gate_filename = _fn 
        gate = np.genfromtxt(gate_filename).astype('float32') 
    if re.search("Freq", _fn):
        spectrum_filename = _fn
        spectrum = np.genfromtxt(spectrum_filename).astype('float32')  

#%%
# DATA MANIPULATION: -----------------------------------------------------------------------------

# Calculate time/delay step and number or steps:
delay_step = np.average(gate[1:len(gate),0]-gate[0:len(gate)-1,0])
delay_dimension = len(gate)
print("Time step: {:.2f} ".format(delay_step), "fs, number of steps: {:.2f} ".format(delay_dimension))

# Calculate frequency step:
frequency_step = np.average(spectrum[1:len(spectrum),0]-spectrum[0:len(spectrum)-1,0])
central_frequency = c/central_wavelength
wavelength_step = c/(central_frequency - (frequency_step/2)*10**15) - c/(central_frequency + (frequency_step/2)*10**15)
frequency_dimension = len(spectrum)
print("Frequency step: {:.6f} ".format(frequency_step), "PHz or {:.3f} ".format(wavelength_step*10**9), "nm, number of steps: {:.2f} ".format(frequency_dimension))

# Generating constant wavelength scale:
if Constant_wavelength:
    wavelength_inconstant = np.zeros(frequency_dimension)  # Initialize empty array
    pixel_of_central_wavelength = np.argmin(abs(spectrum[:,0]))  # Find the pixel of central frequency
    wavelength_inconstant = c/(spectrum[:,0]*10**15+central_frequency)  # Convert frequency scale into wavelength

    # We create constant wavelength, that we will convert into not constant frequency on which we will do interpolation:
    wavelength_constant = np.linspace(wavelength_inconstant[0]*10**9, wavelength_inconstant[-1]*10**9, len(wavelength_inconstant))
    wavelength_constant_step = -np.average(wavelength_constant[1:len(wavelength_constant)]-wavelength_constant[0:len(wavelength_constant)-1])
    print("New wavelength step:  {:.3f}".format(wavelength_constant_step), "nm, number of steps: {:.2f} ".format(len(wavelength_constant)))
    
    # This we will use for interpolation:
    frequency_inconstant = c/(wavelength_constant*10**(-9))

    # If we want to flip the spectrum scare upside down:
    if Flip_up_down:
        wavelength_constant = np.flipud(wavelength_constant)

# If we want to resample time scale:
    if Time_scale_resample:
        time = np.arange(gate[0,0], gate[-1,0], time_step)
        # Recalculating time and wavelength spacing:
        time_step = np.average(time[1:len(time),0]-time[0:len(time)-1,0])
        print("Updated time step: {:.2f} ".format(time_step), "fs, number of steps: {:.2f} ".format(time_step))
        
    else:
        time = gate[:,0]
        time_step = delay_step


# Normalize scales to the interval between -1 and 1 (this is needed for interpolation):
frequency_inconstant_centered = torch.tensor((frequency_inconstant - central_frequency).copy(), dtype=torch.float32)
frequency_inconstant_normalized = frequency_inconstant_centered/torch.max(frequency_inconstant_centered)

time_centered = torch.tensor((time - time[int(len(time)/2)]).copy(), dtype=torch.float32)
time_normalized = time_centered/torch.max(time_centered)

# Stacking together time and wavelength scale:
time_tensor, frequency_tensor = torch.meshgrid(time_normalized, frequency_inconstant_normalized, indexing="xy")

# Creating a grid with these tensors (basically two columns for time coordinate and wavelength coordinate)
grid = torch.stack((time_tensor, frequency_tensor), -1).unsqueeze(0)  # Adding additional dimension, which is needed for torch grid_sample function

# Add more dimensions for frog trace, which is needed for the interpolation that I use:
frog_trace_old = torch.tensor(frog_trace.copy(), dtype = torch.float32).unsqueeze(0).unsqueeze(0)   

# Do the interpolation (align_corners to take pixel corners and not centers)
frog_trace_interp = torch.nn.functional.grid_sample(frog_trace_old, grid, align_corners=True) 

# Go back to original dimensions:
frog_trace_old = frog_trace_old.squeeze(0).squeeze(0)
frog_trace_interp = frog_trace_interp.squeeze(0).squeeze(0)

#%%
# PLOTTING ---------------------------------------------------------------------------------------------

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Time step in plotting
time_step_plt = 300 # in fs
time_interval = int(time_step_plt//time_step)

# Frequency step for plotting:
wavelength_step_plt = 20 # in fs
wavelength_interval = int(wavelength_step_plt//(wavelength_step*10**(9)))

# Plotting part:
data1 = frog_trace_old
data2 = frog_trace_interp

# Figuring out common colorbar:
# find minimum of minima & maximum of maxima
minmin = np.min([torch.min(data1), torch.min(data2)])
maxmax = np.max([torch.max(data1), torch.max(data2)])

# Plotting in two subplots:
fig, axes = plt.subplots(nrows=1, ncols=2) 

fig.tight_layout(pad=2.0)
im1 = axes[0].imshow(data1, vmin=minmin, vmax=maxmax, aspect='equal')
axes[0].set_title("Original FROG trace") 
axes[0].set_xlabel("Delay (fs)")
axes[0].set_ylabel("Wavelength (nm)")
axes[0].set(xticks=np.arange(0,512)[::time_interval], xticklabels=np.array(time_centered[::time_interval]).astype(int))
axes[0].set(yticks=np.arange(0,512)[::wavelength_interval], yticklabels=np.array(wavelength_inconstant[::wavelength_interval]*10**(9)).astype(int))
im2 = axes[1].imshow(data2, vmin=minmin, vmax=maxmax, aspect='equal')
axes[1].set_title("Retrieved FROG trace")
axes[1].set_xlabel("Delay (fs)")
axes[1].set_ylabel("Wavelength (nm)")
axes[1].set(xticks=np.arange(0,512)[::time_interval], xticklabels=np.array(time_centered[::time_interval]).astype(int))
axes[1].set(yticks=np.arange(0,512)[::wavelength_interval], yticklabels=np.array(wavelength_constant[::wavelength_interval]).astype(int))

# Placing colorbar in its own axis:
ax = fig.add_axes([0.2, 0.14, 0.6, 0.03])
fig.colorbar(im2, cax=ax, orientation = 'horizontal')


#%%
# GENERATING HEADER -------------------------------------------------------------------------------------

#%%
# SAVING NEW FROG TRACE ---------------------------------------------------------------------------------
#frog_trace_interp = np.flipud(frog_trace_interp)
fileID = frog_trace_filename[:-4] + "_post_process.txt"
np.savetxt(fileID, frog_trace_interp, fmt='%.3f')


# %%
