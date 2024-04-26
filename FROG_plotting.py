#%%

central_wavelength = 515*10**(-9) #m
c = 299792458 # m/s

# LOAD MEASUREMENT FILES --------------------------------------------------------------------------

import torch
import matplotlib.pyplot as plt
import numpy as np
import re # Used to check if a string contains the specified search pattern
from tkinter.filedialog import askopenfilename  # Module that creates open dialog and return the opened file object in read-only mode

# Select three files:
filenames = askopenfilename(title="Select FROG trace, reconstructed trace, Gate file, and Frequency file", multiple = True)

# Load files based on their names:
for _fn in filenames:  # Iterate through our tuple, which contains path+filename
    if re.search("FROG", _fn):  # Searches in the string for the word and returns True if it finds it
        frog_trace_filename = _fn  # Save file path+name
        frog_trace = np.genfromtxt(frog_trace_filename).astype('float32')  # Open the file as float 32
        frog_trace = np.flipud(frog_trace)
        frog_trace = torch.tensor(frog_trace.copy(), dtype = torch.float32)
    if re.search("Gate", _fn):
        gate_filename = _fn 
        gate = np.genfromtxt(gate_filename).astype('float32')
        time = gate[:,0] 
    if re.search("Freq", _fn):
        spectrum_filename = _fn
        spectrum = np.genfromtxt(spectrum_filename).astype('float32') 
    if re.search("Retrieved", _fn):
        frog_trace_retrieved_filename = _fn  # Save file path+name
        frog_trace_retrieved = np.genfromtxt(frog_trace_retrieved_filename).astype('float32')  # Open the file as float 32
        frog_trace_retrieved = np.flipud(frog_trace_retrieved)
        frog_trace_retrieved = torch.tensor(frog_trace_retrieved.copy(), dtype = torch.float32)

#%%
# DATA ACQUISITION: -----------------------------------------------------------------------------

# Calculate time/delay step and number or steps:
delay_step = np.average(gate[1:len(gate),0]-gate[0:len(gate)-1,0])
delay_dimension = len(gate)
print("Time step: {:.2f} ".format(delay_step), "fs, number of steps: {:.2f} ".format(delay_dimension))

# Calculate frequency step:
frequency_step = np.average(spectrum[1:len(spectrum),0]-spectrum[0:len(spectrum)-1,0])
central_frequency = c/central_wavelength
wavelength_step = c/(central_frequency - (frequency_step/2)*10**15) - c/(central_frequency + (frequency_step/2)*10**15)
frequency_dimension = len(spectrum)
print("Wavelength step: {:.3f} ".format(wavelength_step*10**9), "nm, number of steps: {:.2f} ".format(frequency_dimension))

# Convert frequency scale into wavelength scale:
wavelength_inconstant = c/(spectrum[:,0]*10**15+central_frequency)  
# Center the time scale:
time_centered = torch.tensor((time - time[int(len(time)/2)]).copy(), dtype=torch.float32)
#%%
# PLOTTING ---------------------------------------------------------------------------------------------
Scaling = True

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Defining how much to zoom in the graphs (it will reduce graph by this amount):
if Scaling:
    start_fs = 400 # shift in fs
    end_fs = 400 # shift in fs
    start_nm = 50 # shift in nm
    end_nm = 50 # shift in nm
else:
    start_fs = 0 # shift in fs
    end_fs = 5 # shift in fs
    start_nm = 0 # shift in nm
    end_nm = 1 # shift in nm

# Time step in plotting for axes:
time_step_plt = 200 # in fs

# Frequency step for plotting for axes:
wavelength_step_plt = 20 # in nm


# Plotting part:
data1 = frog_trace/torch.max(frog_trace)
data2 = frog_trace_retrieved/torch.max(frog_trace_retrieved)

# Zooming in the graphs:
start_delay = int(start_fs//delay_step)
end_delay = int(end_fs//delay_step)
start_wavelength = int(start_nm//(wavelength_step*10**9))
end_wavelength = int(end_nm//(wavelength_step*10**9))

#Defining axes spacing:
time_interval = int(time_step_plt//delay_step)
wavelength_interval = int(wavelength_step_plt//(wavelength_step*10**(9)))

# Figuring out common colorbar:
# find minimum of minima & maximum of maxima
minmin = np.min([torch.min(data1), torch.min(data2)])
maxmax = np.max([torch.max(data1), torch.max(data2)])

# Plotting in two subplots:
fig, axes = plt.subplots(nrows=1, ncols=2) 

fig.tight_layout(pad=2.0)
im1 = axes[0].imshow(data1[start_wavelength:-end_wavelength,start_delay:-end_delay], vmin=minmin, vmax=maxmax, aspect='equal')
axes[0].set_title("Original FROG trace") 
axes[0].set_xlabel("Delay (fs)")
axes[0].set_ylabel("Wavelength (nm)")
axes[0].set(xticks=np.arange(0,len(data1))[0:-end_delay-start_delay:time_interval], xticklabels=np.array(time_centered[start_delay:-end_delay:time_interval]).astype(int))
axes[0].set(yticks=np.arange(0,len(data1))[0:-end_wavelength-start_wavelength:wavelength_interval], yticklabels=np.array(wavelength_inconstant[start_wavelength:-end_wavelength:wavelength_interval]*10**(9)).astype(int))

im2 = axes[1].imshow(data2[start_wavelength:-end_wavelength,start_delay:-end_delay], vmin=minmin, vmax=maxmax, aspect='equal')
axes[1].set_title("Retrieved FROG trace") 
axes[1].set_xlabel("Delay (fs)")
axes[1].set_ylabel("Wavelength (nm)")
axes[1].set(xticks=np.arange(0,len(data2))[0:-end_delay-start_delay:time_interval], xticklabels=np.array(time_centered[start_delay:-end_delay:time_interval]).astype(int))
axes[1].set(yticks=np.arange(0,len(data2))[0:-end_wavelength-start_wavelength:wavelength_interval], yticklabels=np.array(wavelength_inconstant[start_wavelength:-end_wavelength:wavelength_interval]*10**(9)).astype(int))

# Placing colorbar in its own axis:
ax = fig.add_axes([0.2, 0.14, 0.6, 0.03])
fig.colorbar(im2, cax=ax, orientation = 'horizontal')


# %%
