
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:12:40 2025

@author: Jake Byrne

This algorithm
"""

# %%
"""-----------------USER-SET PARAMETERS-----------------------------------"""

save_plot = False  #Save the plots?
high_pass = False  #Use a high pass filter?
const_fit = True  #Use a constrained ellipse fit in addition to the free (e.g., you have parameters for PA and inc)?   
use_masking_tool = True
crop=True

step = 90  #How many steps to rotate and operate (360 for 1 degree per measurement)
bin_angle = 1 #bin angle
run_length = 3 #The step function criteria (3-5 recommended)

radial_length = 0.13 #How much of the image should the radial cover? (0.5 is the total image, 0.25 for 1/4 of the image)
radial_check = False #Put as True to visually see the radial and detected points for each pass.

"""Primary Thresholds you want to use (The algorithm loops through each)"""
primary_threshold_list = [5] #Change based on what parameter is changing between loops.10,35,45

"""Gaussian Fitting Parameters"""
Gauss = False    #Use gaussian fitting instead of edge detection?
window_size = 15 #Window size of the sliding window
step_size = 1    #How many pixels to skip between iterations as the sliding window moves (improves speed)
sigma_min, sigma_max = 0.5, 5 #Min and max sigma values to allow for the Gaussian

""" Removal zones - Can define up to 3 azimuthal zones to remove from the
results (for problematic regions of the disk edge). Use in order as the code
breaks if 3 is used and 1 is None."""
# --- define post-removal zones ---
post_zones = [
    (130, 150),   # 1
    None,         # 2
    None,         # 3
]


#Changeable Parameters
init_rotation = 0      #Initial rotation of the image.
bridge_thresh = 120    #How many pixels before the 2-step local outlier removal defines a new section azimuthally.
diff = 40 #THIS IS THE ADDITION TO THE STD DEV BETWEEN THEM
#diff_start, diff_finish = 110,359 #Different zone
diff_multi = 0.5
diff_diff = 6

err_thresh = 50
outlier_thresh = 10 #How many pixels away from average to consider an outlier
bin_threshold = 200 #How many pixels away from average (taken from centre) to consider an outlier for the data points
bootstrap_iter = 10000 #How many iterations for the bootstrapping err calc (free ellipse fit)

json_save = False
manual_star_pos = (512.5,512.5)  #Assumes star in centre, update as tuple (x,y) in pixels

# %%

"""Please update the dictionary with information relating to your image and choose your selection"""

# Dataset selection by number
selected = "1"  # Just change this to "2", "3", etc.

# Dataset selection by number
datasets = {
    "1": {
        "File Name": 'TYC_5709-354-1_2025-03-21_Q_phi.fits',
        "vmin":  -4.38338,
        "vmax": 27.395,
        "Distance (Arcsec)": 6.363,
        "Distance (pc)": 137.82,
        "Inc": 45,
        "PA": 3,
        "Pixel Scale": 12.255,
    },
    "2": {
        "File Name": None,
        "vmin": None,
        "vmax": None,
        "Distance (Arcsec)": None,
        "Distance (pc)": None,
        "Inc": None,
        "PA": None,
        "Pixel Scale": None,
    },
    "3": {
        "File Name": None,
        "vmin": None,
        "vmax": None,
        "Distance (Arcsec)": None,
        "Distance (pc)": None,
        "Inc": None,
        "PA": None,
        "Pixel Scale": None,
    },
    "4": {
        "File Name": None,
        "vmin": None,
        "vmax": None,
        "Distance (Arcsec)": None,
        "Distance (pc)": None,
        "Inc": None,
        "PA": None,
        "Pixel Scale": None,
    },
    "5": {
        "File Name": None,
        "vmin": None,
        "vmax": None,
        "Distance (Arcsec)": None,
        "Distance (pc)": None,
        "Inc": None,
        "PA": None,
        "Pixel Scale": None,
    },
    "6": {
        "File Name": None,
        "vmin": None,
        "vmax": None,
        "Distance (Arcsec)": None,
        "Distance (pc)": None,
        "Inc": None,
        "PA": None,
        "Pixel Scale": None,
    },
    "7": {
        "File Name": None,
        "vmin": None,
        "vmax": None,
        "Distance (Arcsec)": None,
        "Distance (pc)": None,
        "Inc": None,
        "PA": None,
        "Pixel Scale": None,
    },
    "8": {
        "File Name": None,
        "vmin": None,
        "vmax": None,
        "Distance (Arcsec)": None,
        "Distance (pc)": None,
        "Inc": None,
        "PA": None,
        "Pixel Scale": None,
    },
    "9": {
        "File Name": None,
        "vmin": None,
        "vmax": None,
        "Distance (Arcsec)": None,
        "Distance (pc)": None,
        "Inc": None,
        "PA": None,
        "Pixel Scale": None,
    },
    "10": {
        "File Name": None,
        "vmin": None,
        "vmax": None,
        "Distance (Arcsec)": None,
        "Distance (pc)": None,
        "Inc": None,
        "PA": None,
        "Pixel Scale": None,
    }}

# %%


from astropy.io import fits
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import map_coordinates
from datetime import datetime
import os
import json
from scipy import ndimage
from scipy.optimize import curve_fit
import copy
from SEEF_mask import run_masking_tool

# %%



def sample_rotated_line(image, center, angle_deg, length):
    """
    Samples pixel values along a line rotated by angle_deg degrees from the horizontal,
    centered at `center`, with half-length = length.
    """
    angle_rad = np.deg2rad(angle_deg)
    x0, y0 = center
    # Define the line in unrotated space
    half_len = length*radial_length
    x = np.linspace(-half_len, 0, length//2)
    y = np.zeros_like(x)

    # Rotate the coordinates
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

    # Translate to center
    x_rot += x0
    y_rot += y0

    # Sample image at these coordinates
    coords = np.vstack([y_rot, x_rot])  # map_coordinates expects (row, col)
    sampled_values = map_coordinates(image, coords, order=3, mode='nearest')

    return x_rot, y_rot, sampled_values

# Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


# --- unpack into your expected variable names ---
post_rem_start, post_rem_finish   = post_zones[0] if post_zones[0] else (None, None)
post_rem_start1, post_rem_finish1 = post_zones[1] if post_zones[1] else (None, None)
post_rem_start2, post_rem_finish2 = post_zones[2] if post_zones[2] else (None, None)

# Load selected dataset
cfg = datasets[selected]
filename = cfg["File Name"]
mas = cfg["Distance (Arcsec)"]
vmax = cfg["vmax"]
vmin = cfg["vmin"]  
pc_given = cfg["Distance (pc)"]
PA = cfg["PA"]
inc = cfg["Inc"]
pix_scale = cfg["Pixel Scale"]

#Fallbacks for if user inputs are not defined, using typical values of 12.251
#for the pixel scale and 100pc for the distance.
if pix_scale == None:
    pix_scale = 12.251
    

if pc_given == None:
    if mas != None:
        mas_arc = mas/1000
        pc_given = 1/mas_arc
    else:
        pc_given = 100

# Open FITS files
hdulist = fits.open(filename)
hdulist2 = fits.open(filename)

# Check if it's a single 2D image
image_data_out = hdulist[0].data
image_data_test_out = np.copy(image_data_out)

coord_divider = 5 #How many binned coords to skip per item to make the resulting saved images clearer.

sing_img_check = 0

#Expand dimensions if its a single image, so its compatible with code.
if image_data_out.ndim == 2:
    image_data_out = np.expand_dims(image_data_out, axis=0)
    image_data_test_out = np.expand_dims(image_data_test_out, axis=0)
    sing_img_check = 1
    
if sing_img_check == 1:
    id_no = 0
else:
    id_no = 1   #Image number from block
    
std_image = np.copy(image_data_out[id_no])

mean, median, std = sigma_clipped_stats(image_data_out[id_no], sigma=3.0, maxiters=5)

mean1, median1, std1 = sigma_clipped_stats(std_image[200:400,200:400], sigma=3.0, maxiters=5)
mean2, median2, std2 = sigma_clipped_stats(std_image[600:800,600:800], sigma=3.0, maxiters=5)
mean3, median3, std3 = sigma_clipped_stats(std_image[200:400,600:800], sigma=3.0, maxiters=5)
mean4, median4, std4 = sigma_clipped_stats(std_image[600:800,200:400], sigma=3.0, maxiters=5)

std_back = (std1 + std2 + std3 + std4)/4

biny_n = "yes" #Bin? yes or no

limits = True
lines = True
xborder, yborder = 50, 50 #How much to increase the plot size in x or y from max/min points.


plot_lines = "Yes" #Yes or No, for the plot lines
binned_outlier_removal = "Yes"

# List of angles to plot
angles_to_plot = []
for i in range(360):
    if i%20 == 0:
        angles_to_plot.append(i)

#High pass filter
if high_pass == True:
    lowpass = ndimage.gaussian_filter(image_data_out[id_no], 5)
    image_data_gauss = image_data_out[id_no] - lowpass
    image_data_out[id_no] = image_data_out[id_no] + 2*image_data_gauss
    
mean_intensity = np.mean(image_data_out[id_no])
std_intensity = np.std(image_data_out[id_no])
max_intensity = np.max(image_data_out[id_no])
min_intensity = np.min(image_data_out[id_no])

# %%

if use_masking_tool:
    image_data_out = run_masking_tool(
        image_data_out,
        id_no,
        vmin=vmin,
        vmax=vmax,
        crop=crop,
        hdulist=hdulist
    )


# %%


""" Main Algorithm -------------------------------------------------------"""   

for p in primary_threshold_list:
    
    threshold = p #Threshold * std_dev
    
    # Extract the base name of the image (without extension)
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # Get the current date and time in a specific format
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the folder name by combining the base filename with the current date and time
    if save_plot:
        folder_name = f"{base_filename}_{current_datetime}_{threshold}"

        # Create the directory if it doesn't already exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    
    image_data = np.copy(image_data_out)
    image_data_test = np.copy(image_data_test_out)
        
    hdulist.info()
    
    #Variables + lists initialisations
    count=0
    y_coords=[]
    x_coords=[]
    x_lines=[]
    y_lines=[]
    x_coords2=[]
    y_coords2=[]
    slope_calc=[]
    distancelst = []
    bin_dict = {}
    bin_dict1 ={}
    bin_dict_avg={}
    err_dict = {}
    err_dict_x = {}
    err_dict_y = {}
    step_true_forward = []
    step_true_backward = []
    point_error_x = []
    point_error_y = []
    final_err_lst_x = []
    final_err_lst_y = []
    signal = []
    binlst_left_x, binlst_left_y = [], []
    binlst_right_x, binlst_right_y = [], []
    
    #Initial calculations based on changeable parameters
    rotation = 360
    rotation_step = float(360/step)
    bin_step = bin_angle/rotation_step #Creates the step for the angle
    step_no = int((rotation+1)/rotation_step)
    count=0
    count_test=0
    if manual_star_pos == None:
        central_coord = [image_data[id_no].shape[0]/2,image_data[id_no].shape[1]/2]
    else:
        central_coord = manual_star_pos
    centre = central_coord[1]-1
    keycount=bin_angle
    
    #Rotating the image to a nice start point for the algorithm
    image_data[id_no] = rotate(image_data[id_no],angle=init_rotation, reshape=False)
    image_data_test[id_no] = rotate(image_data_test[id_no],angle=init_rotation, reshape=False)
    
    """
    Edge Detection
    ============================================================================
    """
    for i in range(step_no):
        angle_deg = count
        angle_rad_tmp = np.radians(angle_deg + 180)
        # Continuity filtering
        step_true_forward = []
        point_error_x = []
        point_error_y = []
    
        # Determine threshold for this angle
        try:
            if diff_start <= angle_deg <= diff_finish:
                th = diff_multi * threshold * std
                diff_code = diff_diff
            else:
                th = threshold * std
                diff_code = diff
        except:
            th = threshold * std
            diff_code = diff
            
        # Sample a rotated line across the center
        x_rot, y_rot, sampled_values = sample_rotated_line(
            image=image_data[id_no],
            center=central_coord,
            angle_deg=angle_deg,
            length=image_data.shape[1]  # full row width
        )
    
        
        if Gauss == True:
            
            A_min = th
    
            for j in range(0, len(sampled_values) - window_size, step_size):
                x_window = np.arange(j, j + window_size)
                y_window = sampled_values[j : j + window_size]
                y_window_max = np.max(y_window)
                if y_window_max < A_min:
                    continue
            
    
                try:
                    popt, _ = curve_fit(gaussian, x_window, y_window, p0=[max(y_window), np.mean(x_window), 5])
                    A, mu, sigma = popt
                    FWHM = abs(2.355 * sigma)
                    #print(f"[Angle {angle_deg}] A={A:.2f}, mu={mu:.2f}, sigma={sigma:.2f}") 
    
                    if A > A_min and sigma_min < sigma < sigma_max:
                        S_N = A/std_back
                        error_tmp = FWHM/S_N
                        error_x = error_tmp * np.cos(angle_rad_tmp)
                        error_y = error_tmp * np.sin(angle_rad_tmp)
                        point_error_x.append(abs(error_x))
                        point_error_y.append(abs(error_y))
                        step_true_forward.append(int(mu))
                        edge_found = True
                        break  # Break on first valid detection
                except Exception:
                    continue
        else:
            # Threshold
            a_cut = np.where(sampled_values > th)[0]
            if a_cut.size == 0:
                count += rotation_step
                if i % bin_step == 0:
                    keycount += bin_angle
                continue
                
            try:
                for k in range(len(a_cut) - run_length + 1):
                    if np.all(a_cut[k:k + run_length] == np.arange(a_cut[k], a_cut[k] + run_length)):
                        indices = a_cut[k:k + run_length]
            
                        # Use the sampled intensity values
                        y_vals = sampled_values[indices]
                        x_vals = indices  # these are sample indices; you could use actual distances instead
            
                        # 1. Local gradients
                        gradients = np.gradient(y_vals)
                        max_grad = np.max(np.abs(gradients))  # strongest local slope
            
                        # 2. Average gradient across the run
                        delta_y = y_vals[-1] - y_vals[0]
                        delta_x = x_vals[-1] - x_vals[0]
                        avg_grad = delta_y / delta_x if delta_x != 0 else np.nan
                        
                        delta_r = (3*std)/abs(avg_grad)
                        
                        error_x = abs(delta_r*np.cos(angle_rad_tmp))
                        error_y = abs(delta_r * np.sin(angle_rad_tmp))
                        point_error_x.append(error_x)
                        point_error_y.append(error_y)
            
                        step_true_forward.append(a_cut[k])  # or a[k + run_length // 2] for center of edge
                        break
    
            except:
                pass
    
        origin_points = []
        try:
            origin_points.append(step_true_forward[0])
            signal.append(sampled_values[step_true_forward[0]])
        except:
            count += rotation_step
            if i % bin_step == 0:
                keycount += bin_angle
            continue
            
        origin_points = np.array(origin_points)
        
        # Get corresponding rotated coordinates
        x_coords_rot = x_rot[origin_points]
        y_coords_rot = y_rot[origin_points]
        
        if radial_check == True:
            plt.imshow(image_data_test[id_no], vmax=vmax, vmin=vmin, origin='lower', cmap='grey')
            plt.plot(x_rot, y_rot, 'o', color = 'red', markersize=1, label='Sampled Line')
            plt.plot(x_coords_rot, y_coords_rot,'o', color='orange', markersize=3, label="Detected Point")
            plt.legend()
            plt.show()
            
        if angle_deg % 10 == 0:
            print("Computing", angle_deg, "degrees!")
        count += rotation_step
        count_rad = np.deg2rad(count)
    
        if biny_n == "yes":
            x_coords2.extend(x_coords_rot)
            y_coords2.extend(y_coords_rot)
            
        # Binning logic (same as before)
        if i % bin_step == 0 or i == 0:
            try:
                bin_dict1[str(keycount-bin_angle)+"x"] = binlst_left_x
                bin_dict1[str(keycount-bin_angle)+"y"] = binlst_left_y
                err_dict_x[str(keycount-bin_angle)] = point_error_x
                err_dict_y[str(keycount-bin_angle)] = point_error_y
                keycount += bin_angle
            except NameError:
                print("Not created yet")
    
            binlst_left_x, binlst_left_y = [], []
            binlst_right_x, binlst_right_y = [], []

        binlst_left_x.append(x_coords_rot[0])
        binlst_left_y.append(y_coords_rot[0])   
        
    #Adding to the dictionary for the final iteration that does not get captured within the loop
    bin_dict1[str(keycount-10)+"x"] = binlst_left_x
    bin_dict1[str(keycount-10)+"y"] = binlst_left_y
    err_dict_x[str(keycount-10)] = point_error_x
    err_dict_y[str(keycount-10)] = point_error_y
    
    #Rotate image back to original plane
    image_data[id_no] = rotate(image_data[id_no],angle=360-init_rotation, reshape=False)

# %%
    bin_dict = copy.deepcopy(bin_dict1)
    err_dict_x1 = copy.deepcopy(err_dict_x)
    err_dict_y1 = copy.deepcopy(err_dict_y)

    """
    Outlier removal
    =========================================================================
    """
    # Create a list to hold keys that correspond to outliers and need to be deleted
    del_lst = []
   
    def remove_bins(remove_bin_list):
        """
        Removes the bins that are specified in remove_bin_list from the bin_dict.
        """
        for i in remove_bin_list:
            try:
                del bin_dict[i]
                del err_dict_x1[i[:-1]]
                del err_dict_y1[i[:-1]]
            except:
                continue
    
    def get_numeric_part(key):
        """
        Extracts the numeric part of the key (removes the 'x' or 'y' suffix).
        """
        return int(key[:-1])
    
    def remove_bins_in_degree_range(start_deg, end_deg):
        """
        Removes bins within the given degree range from the bin_dict.
        
        Parameters:
        - start_deg: Starting degree (inclusive)
        - end_deg: Ending degree (inclusive)
        """
        # Create a list to hold the keys that should be removed
        remove_bin_list = []
        
        # Iterate over the dictionary keys
        for key in bin_dict.keys():
            # Extract the numeric part of the key
            numeric_part = get_numeric_part(key)
            
            # Check if the numeric part lies within the specified degree range
            if start_deg <= numeric_part <= end_deg:
                remove_bin_list.append(key)
        
        # Call remove_bins function to delete these keys from the dictionary
        remove_bins(remove_bin_list)
        
    def middle_80_avg(arr):
        arr = np.sort(arr)
        n = len(arr)
        k = int(n * 0.2)  # 10% trimmed from each end
        middle_values = arr[k : n - k]
        return np.mean(middle_values)
    
    def rotate_coords_about_center(x, y, angle_deg, center):
        """
        Rotates coordinate arrays (x, y) by angle_deg degrees counterclockwise around a center point.
        """
        angle_rad = np.deg2rad(angle_deg)
        x0, y0 = center
    
        # Shift to center
        x_shifted = x - x0
        y_shifted = y - y0
    
        # Rotate
        x_rot = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
        y_rot = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
    
        # Shift back
        x_final = x_rot + x0
        y_final = y_rot + y0
    
        return x_final, y_final
    
    #Removes empty dict keys
    zero_div_lst = []
    for key,value in bin_dict.items():
        if len(value) == 0:
            zero_div_lst.append(key)
            if 'x' in key:
                zero_div_lst.append(key[:-1]+'y')
            if 'y' in key:
                zero_div_lst.append(key[:-1]+'x')
    remove_bins(zero_div_lst)
            

    #Removing outliers
    for key, value in bin_dict.items():
        avg_value_tmp = np.mean(value)
    
    	#Basic major single point outlier removal, if outlier > average distance to centre + a given threshold, of a given bin, remove it.
        for j in value:
            if abs(avg_value_tmp-j)>outlier_thresh:              
                if 'x' in key:
                    t=bin_dict[key[:-1]+'y'].pop(bin_dict[key].index(j))
                  
                if 'y' in key:
                    t=bin_dict[key[:-1]+'x'].pop(bin_dict[key].index(j))
                        
                r=bin_dict[key].pop(bin_dict[key].index(j))
                
    #Dictionary operation
    #Grabs the first value in each bin for the line segments and averages each for later.
    tmp=[]
    for key, value in bin_dict.items():
        if 'x' in key:
            x_lines.append(value[0])
        if 'y' in key:
            y_lines.append(value[0])
        
        avg_value = sum(value)/len(value)
        bin_dict.update({key:avg_value})
        
    for key, value in err_dict_x1.items():
        try:
            avg_err = sum(value)/len(value)
        except:
            avg_err = 0
            
        err_dict_x1.update({key:avg_err})
        
    for key, value in err_dict_y1.items():
        try:
            avg_err = sum(value)/len(value)
        except:
            avg_err = 0
        err_dict_y1.update({key:avg_err})
        
    #Getting the distance to centre of each bin (average)
    for key, value in bin_dict.items():
        # If the key corresponds to an 'x' coordinate
        if 'x' in key:
            # Calculate the distance from the central coordinate using the Pythagorean theorem
            distance = np.sqrt((value-central_coord[0])**2+(bin_dict[key[:-1]+'y']-central_coord[1])**2)
            # Append the distance to the distance list for further analysis
            distancelst.append(distance)
        # If the key doesn't contain 'x', skip the iteration
        else:
            continue
        
    # Calculate the average distance from the center for all the x, y points
    distance_avg = sum(distancelst)/len(distancelst)
    
    # Loop through each key-value pair again to identify outliers that are extreme.
    for key, value in bin_dict.items():
        # Check only 'x' coordinates for outliers
        if 'x' in key:
            # Calculate the distance from the central coordinate again for comparison
            distance = np.sqrt((value-central_coord[0])**2+(bin_dict[key[:-1]+'y']-central_coord[1])**2)
            # If the distance is greater than the average plus a defined threshold, mark it for deletion
            if distance > distance_avg + bin_threshold:
                # Add both the x and corresponding y key to the list of items to delete
                del_lst.append(key)
                del_lst.append(key[:-1]+'y')
        # Skip any key that isn't an 'x' coordinate
            if np.sqrt(err_dict_x1[key[:-1]]**2+err_dict_y1[key[:-1]]**2) > err_thresh:
                del_lst.append(key)
                del_lst.append(key[:-1]+'y')
                pass
        else:
            continue
    #Removes the accumlated bin keys to delete
    if binned_outlier_removal == "Yes":
        remove_bins(del_lst)
        
    sorted_bin_dict = dict(sorted(bin_dict.items(), key=lambda item: get_numeric_part(item[0])))
    
    tmp = []
    count = 0
    bridge = 0
    new_sect = 0
    new_sect_name = 0
    section_collector = {}
    dict_name = f"sect_{new_sect_name}"
    section_collector[dict_name] = {}
    
        
    for key, value in sorted_bin_dict.items():
        if 'x' in key:
            if new_sect == 1:
                dict_name = f"sect_{new_sect_name}"
                section_collector[dict_name] = {}
                new_sect=0
            if count < 10:
                section_collector[dict_name][key] = distance
                tmp.append(np.sqrt((value-central_coord[0])**2+(bin_dict[key[:-1]+'y']-central_coord[1])**2))
                count = count + 1
                continue
                
            distance = np.sqrt((value-central_coord[0])**2+(bin_dict[key[:-1]+'y']-central_coord[1])**2)
            std_dev = np.std(tmp)
            tmp_distance_avg = middle_80_avg(sorted(tmp))
            
            if distance > tmp_distance_avg + diff_code or distance < tmp_distance_avg - diff_code:
                del_lst.append(key)
                del_lst.append(key[:-1]+'y')
                bridge = bridge + 1
                if bridge == bridge_thresh:
                    bridge = 0
                    count = 0
                    tmp = []
                    new_sect = 1
                    new_sect_name = new_sect_name + 1
                try:
                    if int(key[:-1]) == diff_start or int(key[:-1]) == diff_finish:
                        bridge = 0
                        count = 0
                        tmp = []
                        new_sect = 1
                        new_sect_name = new_sect_name + 1
                except:
                    pass
                continue
            else:
                section_collector[dict_name][key] = distance
                tmp.append(distance)
                bridge = 0
                del tmp[0]
                
    #removing the first "main" section from this outlier removal section.
    removed_dict = section_collector.pop('sect_0')
    
    
    
    tmp = []
    count = 0
    
    #Going over each section (created in the previous outlier removal) and removing bin points beyond the std dev of the rest.
    for dict_name, inner_dict in section_collector.items():
        for key, value in inner_dict.items():  # Loop through the inner dictionary
            tmp.append(value)
            count = count + 1
            
            distance = value      
            tmp_distance_avg = sum(tmp)/len(tmp)
            
        std_dev1 = np.std(tmp)     
        for key, value in inner_dict.items():
            if value > tmp_distance_avg + 1.5*std_dev1 or value < tmp_distance_avg - 1.5*std_dev1:
                del_lst.append(key)
                del_lst.append(key[:-1]+'y')
            
    #Removes the accumlated bin keys to delete
    if binned_outlier_removal == "Yes":
        remove_bins(del_lst)  
        
        # Remove:
        try:
            remove_bins_in_degree_range(post_rem_start, post_rem_finish)  # Remove bins from 30 to 60 degrees
            remove_bins_in_degree_range(post_rem_start1, post_rem_finish1)
            remove_bins_in_degree_range(post_rem_start2, post_rem_finish2)
        except:
            pass
    
    # Takes the values from the bins after they have been averaged for plotting
    for key, value in bin_dict.items():
        
        # If the key corresponds to 'x' coordinates
        if 'x' in key:
            # Check if binning is enabled ('yes')
            if biny_n == "yes":
                # Append the averaged x-coordinate to the list for later plotting
                x_coords.append(value)
              
    
        # If the key corresponds to 'y' coordinates
        if 'y' in key:
            # Check if binning is enabled ('yes')
            if biny_n == "yes":
                # Append the averaged y-coordinate to the list for later plotting
                y_coords.append(value)
              
    
        # If the key corresponds to 'x' coordinates, find the min and max x-values for setting plot limits
        if 'x' in key:
            # Attempt to compare and store the maximum and minimum x values for plotting
            try:
                # Update max_x if the current value is larger
                if value > max_x:
                    max_x = value
                # Update min_x if the current value is smaller
                if value < min_x:
                    min_x = value
            # If this is the first iteration (i.e., no max_x/min_x set), initialize them
            except:
                max_x = value
                min_x = value
    
        # If the key corresponds to 'y' coordinates, find the min and max y-values for setting plot limits
        if 'y' in key:
            # Attempt to compare and store the maximum and minimum y values for plotting
            try:
                # Update max_y if the current value is larger
                if value > max_y:
                    max_y = value
                # Update min_y if the current value is smaller
                if value < min_y:
                    min_y = value
            # If this is the first iteration (i.e., no max_y/min_y set), initialize them
            except:
                max_y = value
                min_y = value
         
    #Adding the errors from the dictionary to a list
    for key, value in err_dict_x1.items():
        final_err_lst_x.append(value)
        
    for key, value in err_dict_y1.items():
        final_err_lst_y.append(value)
            
# %%
    #Converting these to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    x_coords2 = np.array(x_coords2)
    y_coords2 = np.array(y_coords2)
    
    rotation_angle = init_rotation  # or whatever your angle is

    x_coords_final, y_coords_final = rotate_coords_about_center(x_coords, y_coords, rotation_angle, central_coord)
    x_coords2_final, y_coords2_final = rotate_coords_about_center(x_coords2, y_coords2, rotation_angle, central_coord)
    
    x_coords2_final = np.array(x_coords2_final)
    y_coords2_final = np.array(y_coords2_final)
    x_coords2_final = np.array(x_coords2_final)
    y_coords2_final = np.array(y_coords2_final)
    
    signal = np.array(signal)
    S_N_img = np.mean(signal)/std_back
    




    
        
    """
    
    ELLIPSE FITTING
    ==========================================================================================================
    
    """ 
    
    
    def fit_ellipse(x, y):
        """
    
        Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
        the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
        arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].
    
        Based on the algorithm of Halir and Flusser, "Numerically stable direct
        least squares fitting of ellipses'.
    
    
        """
    
        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        return np.concatenate((ak, T @ ak)).ravel() 
    
    def cart_to_pol(coeffs):
        """
    
        Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
        ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
        The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
        ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
        respectively; e is the eccentricity; and phi is the rotation of the semi-
        major axis from the x-axis.
    
        """
    
        # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
        # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
        # Therefore, rename and scale b, d and f appropriately.
        a = coeffs[0]
        b = coeffs[1] / 2
        c = coeffs[2]
        d = coeffs[3] / 2
        f = coeffs[4] / 2
        g = coeffs[5]
    
        den = b**2 - a*c
        if den > 0:
            raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                             ' be negative!')
    
        # The location of the ellipse centre.
        x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den
    
        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))
    
        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp:
            width_gt_height = False
            ap, bp = bp, ap
    
        # The eccentricity.
        r = (bp/ap)**2
        if r > 1:
            r = 1/r
        e = np.sqrt(1 - r)
    
        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi/2
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2
            if a > c:
                phi += np.pi/2
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi/2
        phi = phi % np.pi
    
        return x0, y0, ap, bp, e, phi
 
    def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
        """
        Return npts points on the ellipse described by the params = x0, y0, ap,
        bp, e, phi for values of the parametric variable t between tmin and tmax.
    
        """
    
        x0, y0, ap, bp, e, phi = params
        # A grid of the parametric variable, t.
        t = np.linspace(tmin, tmax, npts)
        x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        return x, y

    def bootstrap_ellipse(x, y, n_iterations=10000):
        params_list = []
    
        for _ in range(n_iterations):
            # Resample with replacement
            indices = np.random.randint(0, len(x), len(x))
            x_sample = x[indices]
            y_sample = y[indices]
    
            try:
                coeffs = fit_ellipse(x_sample, y_sample)
                params = cart_to_pol(coeffs)
                params_list.append(params)
            except Exception as e:
                continue  # Skip failed fits
    
        params_array = np.array(params_list)
        mean_params = np.mean(params_array, axis=0)
        std_errors = np.std(params_array, axis=0)
    
        return mean_params, std_errors
    
    """Free ellipse -------------------------------------------------------------"""
    
    coeffs = fit_ellipse(x_coords_final, y_coords_final)
    
    print('\n')
    print('Free Fitted parameters:') 
    print('-------------------------')
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
    print(f"x center       = {x0:.3f} px")
    print(f"y center       = {y0:.3f} px")
    print(f"semi-major axis = {ap:.3f} px")
    print(f"semi-minor axis = {bp:.3f} px")
    print(f"eccentricity    = {e:.5f}")
    print(f"position angle  = {phi:.2f}°")
  
    x, y = get_ellipse_pts((x0, y0, ap, bp, e, phi))
    
    mean_params_free, std_errors_free = bootstrap_ellipse(x_coords_final, y_coords_final, n_iterations=bootstrap_iter)
    
    # Extract free ellipse params and errors
    x0_f, y0_f, a_f, b_f, e_f, theta_f = mean_params_free
    x0e_f, y0e_f, ae_f, be_f, ee_f, thetae_f = std_errors_free
    
    # Compute percentage errors
    a_fpct = 100 * ae_f / a_f
    b_fpct = 100 * be_f / b_f
    e_fpct = 100 * ee_f / e_f
    theta_fpct = 100 * thetae_f / theta_f
    x0_fpct = 100 * x0e_f / x0_f
    y0_fpct = 100 * y0e_f / y0_f

    
    """ Constrained Ellipse ---------------------------------------------------"""
    
    def fit_ellipse_constr(x, y, PA=None, inc=None):
        """
        Fit an ellipse to a set of 2D points using least squares with optional constraints.
        :param x: 1D array of x coordinates
        :param y: 1D array of y coordinates
        :param PA: Optional fixed position angle in degrees (measured from north, counterclockwise)
        :param inc: Optional fixed inclination in degrees
        :return: Ellipse parameters (a, b, cx, cy, theta)
        """
        # Ensure inputs are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Initial guesses
        cx, cy = centre,centre
        a = np.std(x)
        b = np.std(y) if inc is None else a * np.cos(np.radians(inc))
        theta = 0 if PA is None else np.radians(PA) # Convert PA to ellipse angle
    def ellipse_residuals(params):
        cx, cy, a, b, theta = params
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xt = (x - cx) * cos_t + (y - cy) * sin_t
        yt = -(x - cx) * sin_t + (y - cy) * cos_t
        return (xt**2 / a**2 + yt**2 / b**2 - 1)
    
    def fit_ellipse_constr(x, y, PA=None, inc=None):
        """
        Fit an ellipse to a set of 2D points using least squares with optional constraints.
        :param x: 1D array of x coordinates
        :param y: 1D array of y coordinates
        :param PA: Optional fixed position angle in degrees (measured from north, counterclockwise)
        :param inc: Optional fixed inclination in degrees
        :return: Ellipse parameters (a, b, cx, cy, theta)
        """
        # Ensure inputs are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Initial guesses
        cx, cy = centre,centre
        a = np.std(x)
        b = np.std(y) if inc is None else a * np.cos(np.radians(inc))
        theta = 0 if PA is None else np.radians(PA) # Convert PA to ellipse angle
        
        initial_params = [cx, cy, a, b, theta]
        bounds = ([min(x), min(y), 0, 0, -np.pi], [max(x), max(y), np.inf, np.inf, np.pi])
        result = least_squares(ellipse_residuals, initial_params, bounds=bounds, jac='2-point')
        
        # Calculate the Jacobian matrix of the residuals
        jacobian = result.jac
        
        # Compute covariance matrix using the Jacobian and the residuals
        # Covariance matrix is given by: Cov = (J^T * J)^(-1) * (sigma^2)
        # We approximate the error (variance) as the sum of squared residuals (chi-square)
        residuals = result.fun
        chi_square = np.sum(residuals ** 2) / len(residuals)
        
        # Covariance matrix
        covariance_matrix = np.linalg.inv(np.dot(jacobian.T, jacobian)) * chi_square
        
        # Standard deviations (errors)
        errors = np.sqrt(np.diagonal(covariance_matrix))
        
        return result.x[2], result.x[3], result.x[0], result.x[1], result.x[4], errors  # Return (b, a, cx, cy, theta)
    
    def plot_ellipse(a, b, cx, cy, theta, x, y):
        """ Plot the fitted ellipse along with the data points. """
        t = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = a * np.cos(t)
        ellipse_y = b * np.sin(t)
        
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rotated = cx + ellipse_x * cos_t - ellipse_y * sin_t
        y_rotated = cy + ellipse_x * sin_t + ellipse_y * cos_t
        
        plt.scatter(x, y, marker='.',label="Data Points")
        plt.plot(x_rotated, y_rotated, 'r', label="Fitted Ellipse")
        plt.scatter([cx], [cy], color='g', marker='x', label="Center")
        plt.axis('equal')
        plt.legend(fontsize='xx-small')
        plt.show()
        
        return x_rotated, y_rotated
    
    if const_fit == True:
        ellipse_params = fit_ellipse_constr(x_coords_final, y_coords_final, PA=PA, inc=inc)  # Constrain PA and inclination
        
        # Unpack parameters and errors
        a, b, cx, cy, theta = ellipse_params[:5]
        errors = ellipse_params[5]
        
        # Individual errors
        err_a, err_b, err_cx, err_cy, err_theta = errors
        
        # Percentage errors
        pct_err_a = 100 * abs(err_a / a) if a != 0 else float('inf')
        pct_err_b = 100 * abs(err_b / b) if b != 0 else float('inf')
        pct_err_cx = 100 * abs(err_cx / cx) if cx != 0 else float('inf')
        pct_err_cy = 100 * abs(err_cy / cy) if cy != 0 else float('inf')
        pct_err_theta = 100 * abs(err_theta / theta) if theta != 0 else float('inf')
    
    
        
        x_rotated, y_rotated = plot_ellipse(*ellipse_params[:5], x_coords_final, y_coords_final)
        
        """ Calculating eccentricity for constrained and error"""
        
        # The eccentricity for constrained.
        r_const = (ellipse_params[0]/ellipse_params[1])**2
        if r_const > 1:
            r_const = 1/r_const
        e_const = np.sqrt(1 - r_const)
    
        # Given values
        a = ellipse_params[0]
        b = ellipse_params[1]
        sigma_a = err_a
        sigma_b = err_b
        
        e = e_const
        
        # Partial derivatives
        de_da = (b**2) / (a**3 * e)
        de_db = (-b) / (a**2 * e)
        
        # Error propagation
        sigma_e = np.sqrt((de_da * sigma_a)**2 + (de_db * sigma_b)**2)
        
        # Percentage error
        sigma_e_prct = (sigma_e / e) * 100
        
        theta_deg = np.degrees(theta)
        err_theta_deg = np.degrees(err_theta)
        
        a, b, cx, cy, theta, err= ellipse_params
        if a < b:
            a, b = b, a

        
        print('\n')
        print('Constrained Fitted Parameters')
        print('-------------------------')

        print(f"x center (cx)        = {cx:.3f} px")
        print(f"y center (cy)        = {cy:.3f} px")
        print(f"Semi-major axis (a)  = {a:.3f} px")
        print(f"Semi-minor axis (b)  = {b:.3f} px")
        print(f"Eccentricity (e)     = {e:.5f}")
        print(f"Rotation angle (θ)   = {theta:.2f}°")

    


    
    """
    Height Measurements
    ===============================================================================================================
    """
    
    def calculate_inclination(a, b):
        if a == 0 or b == 0:
            raise ValueError("Semi-major and semi-minor axes must be nonzero.")
        if b > a:
            a, b = b, a
            #raise ValueError("Semi-minor axis cannot be larger than the semi-major axis.")
        
        i = np.arccos(b / a)  # Compute inclination in radians
        return np.degrees(i)  # Convert to degrees
    
    #Free fit
    offset_x_free = x0-centre
    offset_y_free = y0-centre
    
    """Error is same as offset"""
    u_free_pix = np.sqrt(offset_x_free**2 + offset_y_free**2)
    u_free_pix_err = np.sqrt((offset_x_free**2 * x0e_f**2 + offset_y_free**2 * y0e_f**2)) / u_free_pix
    
    u_free = ((u_free_pix * pix_scale)/1000)*pc_given
    u_free_err = ((u_free_pix_err * pix_scale)/1000)*pc_given
    
    i_free_deg = calculate_inclination(ap,bp)
    i_free = calculate_inclination(ap,bp) * np.pi/180
    i_free_err = (np.sqrt((bp * ae_f / ap**2)**2 + (be_f / ap)**2))/np.sin(i_free)
    i_free_err_deg = i_free_err/(np.pi/180)
    
    r_as = (ap * pix_scale)/1000
    r_free = r_as * pc_given
    r_free_err = (pix_scale * pc_given / 1000) * ae_f


    if np.sin(i_free) == 0:  # Avoid division by zero
        raise ValueError("sin(i) cannot be zero")
    else:
        H_free = u_free / np.sin(i_free)
        
        H_free_err = np.sqrt(
            (u_free_err / np.sin(i_free))**2 +
            ((u_free * np.cos(i_free) * i_free_err) / np.sin(i_free)**2)**2
        )
        
    H_r_free = H_free/r_free
    H_r_free_err = np.sqrt(
        (H_free_err / r_free)**2 +
        (H_free * r_free_err / r_free**2)**2
    ) 
    
    if np.sin(i_free) == 0:  # Avoid division by zero
        raise ValueError("sin(i) cannot be zero")
    else:
        H_free = u_free / np.sin(i_free)
        
    # Compute percentage errors
    H_free_err_pct = 100 * H_free_err / H_free
    i_free_err_pct = 100 * i_free_err_deg / i_free_deg
    r_free_err_pct = 100 * r_free_err / r_free
    u_free_err_pct = 100 * u_free_err / u_free
    H_r_free_err_pct = 100 * H_r_free_err / H_r_free
        
    if const_fit == True:
        #Constrained fit
        offset_x_const = ellipse_params[2]-centre
        offset_y_const = ellipse_params[3]-centre
        
        """error is same as offset"""
        u_const_pix = np.sqrt(offset_x_const**2 + offset_y_const**2)
        u_const_pix_err = np.sqrt((offset_x_const**2 * err_cx**2 + offset_y_const**2 * err_cy**2)) / u_free_pix
        
        u_const = ((u_const_pix * pix_scale)/1000)*pc_given
        u_const_err = ((u_const_pix_err * pix_scale)/1000)*pc_given
        
        i_const_deg = calculate_inclination(ellipse_params[0], ellipse_params[1])
        i_const = calculate_inclination(ellipse_params[0], ellipse_params[1]) * np.pi/180
        i_const_err = (np.sqrt((ellipse_params[1] * err_a / ellipse_params[0]**2)**2 + (err_b / ellipse_params[0])**2))/np.sin(i_const)
        i_const_err_deg = i_const_err/(np.pi/180)
        
        if ellipse_params[0] < ellipse_params[1]:
            r_tmp = ellipse_params[1]
        else:
            r_tmp = ellipse_params[0]
        
        r_as = (r_tmp * pix_scale)/1000
        r_const = r_as * pc_given
        r_const_err = (pix_scale * pc_given / 1000) * err_a
    
        
        if np.sin(i_const) == 0:  # Avoid division by zero
            raise ValueError("sin(i) cannot be zero")
        else:
            H_const = u_const / np.sin(i_const)
            
            H_const_err = np.sqrt(
                (u_const_err / np.sin(i_const))**2 +
                ((u_const * np.cos(i_const) * i_const_err) / np.sin(i_const)**2)**2
            )
            
        H_r_const = H_const/r_const
        
        H_r_const_err = np.sqrt(
            (H_const_err / r_const)**2 +
            (H_const * r_const_err / r_const**2)**2
        ) 
        
        H_const_err_pct = 100 * H_const_err / H_const
        i_const_err_pct = 100 * i_const_err_deg / i_const_deg
        r_const_err_pct = 100 * r_const_err / r_const
        u_const_err_pct = 100 * u_const_err / u_const
        H_r_const_err_pct = 100 * H_r_const_err / H_r_const


    if const_fit == True:
        ellipses = {
            "ellipse_free": {"a": ap, "b": bp, "cx": x0, "cy": y0, "theta": phi},
            "ellipse_const": {"a": ellipse_params[0], "b": ellipse_params[1], "cx": ellipse_params[2], "cy": ellipse_params[3], "theta": ellipse_params[4]},
        }
    else:
        ellipses = {
            "ellipse_free": {"a": ap, "b": bp, "cx": x0, "cy": y0, "theta": phi}
       }
    
    
    
    
    
    
# %%


    """
    Final Plotting
    =============================================================================================================
    """
    
    # Function to find the nearest available angle in the dictionary
    def find_nearest_angle(angle, direction=1):
        while True:
            x_key = f"{angle}x"
            y_key = f"{angle}y"
            
            if x_key in sorted_bin_dict and y_key in sorted_bin_dict:
                return x_key, y_key  # Return the found keys
            
            # If not found, try the next angle by incrementing or decrementing
            angle += direction
            if angle >= 360:  # Wrap around if it exceeds 360 degrees
                angle = 0
            elif angle < 0:  # Wrap around if the angle goes below 0
                angle = 359
    
    # Display the image data using imshow
    # vmin and vmax are used to define the color scale range for imshow
    plt.imshow(image_data_test[id_no], vmin=vmin, vmax=vmax)  # The image is displayed with a defined color scale range.
    
    
    # Plot the first set of coordinates (y_coords vs x_coords) in red with very small markers
    #plt.plot(x_coords_final, y_coords, 'o', color = 'red', markersize=3)
    plt.errorbar(x_coords, y_coords, xerr=final_err_lst_x, yerr=final_err_lst_y, fmt='o', color = 'red', markersize=3, capsize=0, elinewidth=0.3, label='Binned data with errors')

    
    # Plot the second set of coordinates (y_coords2 vs x_coords2) in blue with very small markers
    plt.plot(x_coords2, y_coords2, 'b,', markersize=0.1)  # Blue scatter plot for the second set of points, also with tiny markers.
    
    if const_fit == True:
        x_rotated_tmp, y_rotated_tmp = rotate_coords_about_center(x_rotated, y_rotated, -rotation_angle, central_coord)
        plt.plot(x_rotated_tmp, y_rotated_tmp, 'r', markersize=0.05,label="Constrained")
    
    x_tmp, y_tmp = rotate_coords_about_center(x, y, -rotation_angle, central_coord)
    
    # Plot the line defined by (x, y)
    plt.plot(x_tmp, y_tmp, markersize=0.05, label="Free")  # Simple plot for the line connecting the given x and y points.
    
    # Create an empty list to store the points for these angles
    points_to_plot = {}
    
    # Extract the x and y values for the specified angles, trying the nearest available angle if the exact one isn't found
    for angle in angles_to_plot:
        x_key, y_key = find_nearest_angle(angle)
    
        x_value = sorted_bin_dict[x_key]  # Take the first value in each bin (modify if necessary)
        y_value = sorted_bin_dict[y_key]
        
        points_to_plot[angle] = (x_value, y_value)
        
        # Plot the lines from the center to the points at 0°, 90°, 180°, and 270°
    if lines == True:
        for angle, (x_value, y_value) in points_to_plot.items():
            plt.plot([central_coord[0], x_value], [central_coord[1], y_value], color='grey', marker='.', linestyle='dashed', linewidth=0.2, markersize=0.1)
            plt.text((central_coord[0] + x_value) / 2, (central_coord[1] + y_value) / 2, f'{angle}°', fontsize=4, color='black')
    
    
    # Add a colorbar to the plot for the imshow display
    #plt.colorbar()
    if limits == True:
        # Set the x-axis limits for the plot to extend slightly beyond the minimum and maximum x values
        plt.xlim(min_x-xborder, max_x+xborder)
        # Set the y-axis limits for the plot to extend slightly beyond the minimum and maximum y values
        plt.ylim(min_y-yborder, max_y+yborder)
    plt.xlabel("X Pixel")
    plt.ylabel("Y pixel")
    # Move the annotation text outside the plot
    fig = plt.gcf()  # Get the current figure
    
    mse = np.mean((image_data_out[id_no] - image_data[id_no])**2)
    
    if const_fit == False:
        H_const = H_const_err = H_const_err_pct = \
        u_const = u_const_err = u_const_err_pct = \
        i_const_deg = i_const_err_deg = i_const_err_pct = \
        r_const = r_const_err = r_const_err_pct = \
        H_r_const = H_r_const_err = H_r_const_err_pct = \
        e_f = ee_f = e_fpct = \
        cx = err_cx = cy = err_cy = \
        a = err_a = pct_err_a = \
        b = err_b = pct_err_b = \
        theta_deg = err_theta_deg = pct_err_theta = \
        e_const = sigma_e = sigma_e_prct = 0

 
    
    annotation_text = (
        "Main Results\n"
        "--------------------------\n"
        "H_free = {:.3f} ± {:.3f} ({:.1f}%) AU\n"
        "u_free = {:.3f} ± {:.3f} ({:.1f}%) AU\n"
        "i_free = {:.3f} ± {:.3f} ({:.1f}%)°\n"
        "R_free = {:.3f} ± {:.3f} ({:.1f}%) AU\n"
        "H_const = {:.3f} ± {:.3f} ({:.1f}%) AU\n"
        "u_const = {:.3f} ± {:.3f} ({:.1f}%) AU\n"
        "i_const = {:.3f} ± {:.3f} ({:.1f}%)°\n"
        "R_const = {:.3f} ± {:.3f} ({:.1f}%) AU\n"
        "H_R_free = {:.3f} ± {:.3f} ({:.1f}%)\n"
        "H_R_const = {:.3f} ± {:.3f} ({:.1f}%)\n"
        "\n"
        "Parameters"
        "--------------------------\n"
        "High Pass Filter = {}\n"
        "Gaussian Threshold = {}\n"
        "Init Rotation = {}°\n"
        "Bridge Thresh = {}\n"
        "Total steps = {}\n"
        "Bin Angle = {}°\n"
        "Threshold = {}\n"
        "Outlier Thresh = {}\n"
        "Bin Thresh = {}\n"
        "Difference allowance between pixels * std = {}\n"
        "\n"
        "Others"
        "--------------------------\n"
        "Mean (Pixel intensity) = {:.2e}\n"
        "Median (Pixel intensity) = {:.2e}\n"
        "Std Dev (Pixel intensity) = {:.2e}\n"
        "Total MSE after interpolation {:.10f}\n"
        "Average S/N of points {:.2f}\n"
        "\n"
        "Free Ellipse\n"
        "--------------------------\n"
        "Center = ({:.2f} ± {:.2f}, {:.2f} ± {:.2f}) px\n"
        "a (semi-major axis) = {:.3f} ± {:.3f} px ({:.2f}%)\n"
        "b (semi-minor axis) = {:.3f} ± {:.3f} px ({:.2f}%)\n"
        "theta (angle, deg) = {:.3f}° ± {:.3f}° ({:.2f}%)\n"
        "Eccentricity = {:.3f} ± {:.3f} ({:.2f}%)\n"
        
        "\n"
        "Constrained Ellipse\n"
        "--------------------------\n"
        "Center = ({:.2f} ± {:.2f}, {:.2f} ± {:.2f}) px\n"
        "a (semi-major axis): {:.3f} ± {:.3f} ({:.2f}%)\n"
        "b (semi-minor axis): {:.3f} ± {:.3f} ({:.2f}%)\n"
        "theta (angle, deg):  {:.3f} ± {:.3f} ({:.2f}%)\n"
        "Eccentricity = {:.3f} ±  {:.3f} ({:.2f}%)\n"
        "Optional Parameters\n"
        "--------------------------\n"

    ).format(
        H_free, H_free_err, H_free_err_pct, u_free, u_free_err, u_free_err_pct, i_free_deg, i_free_err_deg, i_free_err_pct, r_free, r_free_err, r_free_err_pct,
        H_const, H_const_err, H_const_err_pct, u_const, u_const_err, u_const_err_pct, i_const_deg, i_const_err_deg, i_const_err_pct, r_const, r_const_err, r_const_err_pct,
        H_r_free, H_r_free_err, H_r_free_err_pct, H_r_const, H_r_const_err, H_r_const_err_pct,
        high_pass, Gauss, init_rotation, bridge_thresh, step, bin_angle, threshold, 
        outlier_thresh, bin_threshold, diff, mean, median, std, mse, S_N_img,
        x0_f, x0e_f, y0_f, y0e_f,
        a_f, ae_f, a_fpct,
        b_f, be_f, b_fpct,
        np.degrees(theta_f), np.degrees(thetae_f), theta_fpct,
        e_f, ee_f, e_fpct,
        cx, err_cx, cy, err_cy,
        a, err_a, pct_err_a,
        b, err_b, pct_err_b,
        theta_deg, err_theta_deg, pct_err_theta, e_const, sigma_e, sigma_e_prct
    )

    
    # Optional values to append if they exist
    optional_params = []
    
    # Check and add optional parameters
    for var_name in [
        ("diff_start", "Diff Start"),
        ("diff_finish", "Diff Finish"),
        ("diff_multi", "Diff Multiplier"),
        ("post_rem_start", "Post Removal Start"),
        ("post_rem_finish", "Post Removal Finish"),
        ("post_rem_start1", "Post Removal Start 1"),
        ("post_rem_start1", "Post Removal Start 1"),
        ("post_rem_start2", "Post Removal Start 2"),
        ("post_rem_start2", "Post Removal Start 2"),
        ("pc_given", "Input distance in pc"),
        ("mas", "Input distance in mas"),
        ("PA", "Known PA"),
        ("inc", "Known inclination"),
        ("mm_flux", "mm flux (mJy)"),
        ("H_given", "Known H"),
    ]:
        if var_name[0] in globals():
            optional_params.append(f"{var_name[1]} = {globals()[var_name[0]]}")
    
    # Combine annotation_text with optional ones
    if optional_params:
        annotation_text += "\n\n" + "\n".join(optional_params)

    
    # Add the annotation outside the plot area using fig.text()
    fig.text(0.82, 0.55, annotation_text, ha='left', va='center', fontsize=8, color='black', rotation=0,bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3'))
    plt.legend(loc='upper right', bbox_to_anchor=(1.42, 1))  # Push legend to the right of plot
    #plt.title("Threshold = {} sigma of the BG".format(p))
    plt.show() 
    
    if save_plot == True:
        # Save the annotation text with the new naming convention (based on base_filename and current_datetime)
        annotation_filename = os.path.join(folder_name, f"{base_filename}_{current_datetime}.txt")
        with open(annotation_filename, 'w') as f:
            f.write(annotation_text)
    
    """ coordinate conversion to arcseconds """
    
    pixel_scale_arcsec = pix_scale/1000
    
    final_err_lst_y = np.array(final_err_lst_y)
    final_err_lst_x = np.array(final_err_lst_x)
    
    
    
    """ ALL POINTS ------------------------------------------------"""
    coords_lst_counter=0
    for t in range(2):
        if coords_lst_counter == 1 and save_plot == True:
            x_coords_final = x_coords_final[::coord_divider]
            y_coords_final = y_coords_final[::coord_divider]
            final_err_lst_y = final_err_lst_y[::coord_divider]
            final_err_lst_x = final_err_lst_x[::coord_divider]
            prefolder = 'Coords_Spaced'
            folder_name = os.path.join(folder_name, prefolder)
        else:
            coords_lst_counter += 1
        # Now save the images into the created folder
        if save_plot == True:
            subfolder = "all"
            save_folder = os.path.join(folder_name, subfolder)
            # Create the subfolder if it doesn't exist
            os.makedirs(save_folder, exist_ok=True)
            
            # Save the original image plot
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            
            # Generate tick positions based on min and max values for both axes
            x_ticks = np.linspace(min_x - xborder, max_x + xborder, 7)  # You can adjust number of ticks here
            y_ticks = np.linspace(min_y - yborder, max_y + yborder, 7)
            
            # Convert pixel positions to arcseconds, with center pixel (512, 512) as zero
            x_ticks_arcsec = (x_ticks - centre) * pixel_scale_arcsec
            y_ticks_arcsec = (y_ticks - centre) * pixel_scale_arcsec
            
            # Remove the middle tick and insert zero
            x_ticks = np.delete(x_ticks, len(x_ticks)//2)  # Remove the middle tick
            x_ticks_arcsec = np.delete(x_ticks_arcsec, len(x_ticks_arcsec)//2)  # Remove the middle value
            
            # Do the same for y-axis
            y_ticks = np.delete(y_ticks, len(y_ticks)//2)  # Remove the middle tick
            y_ticks_arcsec = np.delete(y_ticks_arcsec, len(y_ticks_arcsec)//2)  # Remove the middle value
            
            # Manually add 0 for the center tick, ensuring it is labeled as zero
            x_ticks_arcsec = np.concatenate(([0], x_ticks_arcsec))
            x_ticks = np.concatenate(([centre], x_ticks))
            
            y_ticks_arcsec = np.concatenate(([0], y_ticks_arcsec))
            y_ticks = np.concatenate(([centre], y_ticks))
            
            # Apply the new ticks to the plot, rounding for readability
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            
            plt.title("Original Image")
            plt.savefig(os.path.join(save_folder, 'original_image.png'), bbox_inches='tight')
            plt.clf()  # Clear the figure after saving
        
            # Save the plot with the points
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label="Binned")
            plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1, label="Raw")
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Image with data points")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'image_with_points.png'), bbox_inches='tight')
            plt.clf()
        
            # Save the plot with the free ellipse (with points)
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)
            plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label='Binned Data Points')
            plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Free Ellipse w/points")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'free_ellipse_with_points.png'), bbox_inches='tight')
            plt.clf()
        
            if const_fit == True:
                # Save the plot with the constrained ellipse (with points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)
                plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label='Binned Data Points')
                plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Constrained Ellipse w/points")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'constrained_ellipse_with_points.png'), bbox_inches='tight')
                plt.clf()
            
            if const_fit == True:
                # Save the plot with both ellipses (with points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4], zorder=3)
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)
                plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label='Binned Data Points')
                plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Both Ellipses w/points")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'both_ellipses_with_points.png'), bbox_inches='tight')
                plt.clf()
        
            # Save the plot with the free ellipse (without points)
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)  # Only the free ellipse without points
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Free Ellipse")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'free_ellipse_without_points.png'), bbox_inches='tight')
            plt.clf()
            
            if const_fit == True:
                # Save the plot with the constrained ellipse (without points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)  # Only the constrained ellipse without points
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Constrained Ellipse")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'constrained_ellipse_without_points.png'), bbox_inches='tight')
                plt.clf()
                
            if const_fit == True:
                # Save the plot with both ellipses (without points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)  # Both ellipses without points
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Both Ellipse Fits")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'both_ellipses_without_points.png'), bbox_inches='tight')
                plt.clf()
            
            #Only points and ellipses, no image
            plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)
            if const_fit == True:
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)  # Both ellipses with points
            plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label='Binned Data Points')
            plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Both Ellipse Fits")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'both_ellipses_with_points(no_img).png'), bbox_inches='tight')
            
            plt.clf()
            
            import matplotlib.image as mpimg
            
            # Filenames of the individual images
            if const_fit == True:
                image_filenames = [
                    'original_image.png',
                    'image_with_points.png',
                    'free_ellipse_with_points.png',
                    'constrained_ellipse_with_points.png',
                    'both_ellipses_with_points.png',
                    'free_ellipse_without_points.png',
                    'constrained_ellipse_without_points.png',
                    'both_ellipses_without_points.png'
                ]
                
                # Create a figure with subplots (2 rows x 4 columns)
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjust figsize as needed
                
                for ax, img_name in zip(axes.flat, image_filenames):
                    img_path = os.path.join(save_folder, img_name)
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis('off')  # Remove axis and ticks
                    ax.set_title('')  # Just to make sure
                
                # Save the final collage
                combined_path = os.path.join(save_folder, 'ALL.png')
                plt.tight_layout(pad=0.5)
                plt.savefig(combined_path, bbox_inches='tight')
                plt.close()
                
            else:
                image_filenames = [
                    'original_image.png',
                    'image_with_points.png',
                    'free_ellipse_with_points.png',
                    'free_ellipse_without_points.png',
                ]
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # Square aspect, adjust if needed
                
                for ax, img_name in zip(axes.flat, image_filenames):
                    img_path = os.path.join(save_folder, img_name)
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis('off')
                
                combined_path = os.path.join(save_folder, 'ALL.png')
                plt.subplots_adjust(wspace=0, hspace=0)  # No space between subplots
                plt.savefig(combined_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                
            
            
            """
            JSON
            """
            
            # Save path: same directory as your script
            json_file = os.path.join(folder_name, 'saved_ellipses.json')
            
            # Make sure the file exists (create if not)
            if not os.path.exists(json_file):
                with open(json_file, 'w') as f:
                    json.dump({}, f, indent=4)
            
            # Load existing data
            with open(json_file, 'r') as f:
                all_ellipses = json.load(f)
            
            # Save under a unique key
            label = threshold
            all_ellipses[label] = ellipses
            
            # Write updated data back to file
            with open(json_file, 'w') as f:
                json.dump(all_ellipses, f, indent=4)
            
        
        
        """ONLY BINNED POINTS ------------------------------------------- """
        
        # Now save the images into the created folder
        if save_plot:
            subfolder = "binned"
            save_folder = os.path.join(folder_name, subfolder)
            # Create the subfolder if it doesn't exist
            os.makedirs(save_folder, exist_ok=True)
            
            # Save the original image plot
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            
            # Generate tick positions based on min and max values for both axes
            x_ticks = np.linspace(min_x - xborder, max_x + xborder, 7)  # You can adjust number of ticks here
            y_ticks = np.linspace(min_y - yborder, max_y + yborder, 7)
            
            # Convert pixel positions to arcseconds, with center pixel (512, 512) as zero
            x_ticks_arcsec = (x_ticks - centre) * pixel_scale_arcsec
            y_ticks_arcsec = (y_ticks - centre) * pixel_scale_arcsec
            
            # Remove the middle tick and insert zero
            x_ticks = np.delete(x_ticks, len(x_ticks)//2)  # Remove the middle tick
            x_ticks_arcsec = np.delete(x_ticks_arcsec, len(x_ticks_arcsec)//2)  # Remove the middle value
            
            # Do the same for y-axis
            y_ticks = np.delete(y_ticks, len(y_ticks)//2)  # Remove the middle tick
            y_ticks_arcsec = np.delete(y_ticks_arcsec, len(y_ticks_arcsec)//2)  # Remove the middle value
            
            # Manually add 0 for the center tick, ensuring it is labeled as zero
            x_ticks_arcsec = np.concatenate(([0], x_ticks_arcsec))
            x_ticks = np.concatenate(([centre], x_ticks))
            
            y_ticks_arcsec = np.concatenate(([0], y_ticks_arcsec))
            y_ticks = np.concatenate(([centre], y_ticks))
            
            # Apply the new ticks to the plot, rounding for readability
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            
            plt.title("Original Image")
            plt.savefig(os.path.join(save_folder, 'original_image.png'), bbox_inches='tight')
            plt.clf()  # Clear the figure after saving
        
            # Save the plot with the points
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label="Binned")
            #plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1, label="Raw")
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Image with data points")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'image_with_points.png'), bbox_inches='tight')
            plt.clf()
        
            # Save the plot with the free ellipse (with points)
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.',dashes=[4, 2, 1, 4], zorder=3)
            plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label='Binned Data Points')
            #plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Free Ellipse w/points")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'free_ellipse_with_points.png'), bbox_inches='tight')
            plt.clf()
            
            if const_fit == True:
                # Save the plot with the constrained ellipse (with points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)
                plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label='Binned Data Points')
                #plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Constrained Ellipse w/points")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'constrained_ellipse_with_points.png'), bbox_inches='tight')
                plt.clf()
                
                # Save the plot with both ellipses (with points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)
                plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label='Binned Data Points')
                #plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Both Ellipses w/points")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'both_ellipses_with_points.png'), bbox_inches='tight')
                
                plt.clf()
        
            # Save the plot with the free ellipse (without points)
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)  # Only the free ellipse without points
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Free Ellipse")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'free_ellipse_without_points.png'), bbox_inches='tight')
            plt.clf()
            
            if const_fit == True:
                # Save the plot with the constrained ellipse (without points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)  # Only the constrained ellipse without points
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Constrained Ellipse")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'constrained_ellipse_without_points.png'), bbox_inches='tight')
                plt.clf()
        
                # Save the plot with both ellipses (without points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4], zorder=3)
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)  # Both ellipses without points
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Both Ellipse Fits")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'both_ellipses_without_points.png'), bbox_inches='tight')
                plt.clf()
            
            #Only points and ellipses, no image
            plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)
            if const_fit == True:
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)  # Both ellipses with points
            plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label='Binned Data Points')
            plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Both Ellipse Fits")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'both_ellipses_with_points(no_img).png'), bbox_inches='tight')
            
            plt.clf()
            
            # Filenames of the individual images
            if const_fit == True:
                image_filenames = [
                    'original_image.png',
                    'image_with_points.png',
                    'free_ellipse_with_points.png',
                    'constrained_ellipse_with_points.png',
                    'both_ellipses_with_points.png',
                    'free_ellipse_without_points.png',
                    'constrained_ellipse_without_points.png',
                    'both_ellipses_without_points.png'
                ]
                
                # Create a figure with subplots (2 rows x 4 columns)
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjust figsize as needed
                
                for ax, img_name in zip(axes.flat, image_filenames):
                    img_path = os.path.join(save_folder, img_name)
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis('off')  # Remove axis and ticks
                    ax.set_title('')  # Just to make sure
                
                # Save the final collage
                combined_path = os.path.join(save_folder, 'ALL.png')
                plt.tight_layout(pad=0.5)
                plt.savefig(combined_path, bbox_inches='tight')
                plt.close()
                
            else:
                image_filenames = [
                    'original_image.png',
                    'image_with_points.png',
                    'free_ellipse_with_points.png',
                    'free_ellipse_without_points.png',
                ]
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # Square aspect, adjust if needed
                
                for ax, img_name in zip(axes.flat, image_filenames):
                    img_path = os.path.join(save_folder, img_name)
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis('off')
                
                combined_path = os.path.join(save_folder, 'ALL.png')
                plt.subplots_adjust(wspace=0, hspace=0)  # No space between subplots
                plt.savefig(combined_path, bbox_inches='tight', pad_inches=0)
                plt.close()

    
                
        """ONLY BINNED POINTS WITH ERROR ---------------------------------- """
        
        # Now save the images into the created folder
        if save_plot:
            subfolder = "error"
            save_folder = os.path.join(folder_name, subfolder)
            # Create the subfolder if it doesn't exist
            os.makedirs(save_folder, exist_ok=True)
            # Save the original image plot
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            
            # Generate tick positions based on min and max values for both axes
            x_ticks = np.linspace(min_x - xborder, max_x + xborder, 7)  # You can adjust number of ticks here
            y_ticks = np.linspace(min_y - yborder, max_y + yborder, 7)
            
            # Convert pixel positions to arcseconds, with center pixel (512, 512) as zero
            x_ticks_arcsec = (x_ticks - centre) * pixel_scale_arcsec
            y_ticks_arcsec = (y_ticks - centre) * pixel_scale_arcsec
            
            # Remove the middle tick and insert zero
            x_ticks = np.delete(x_ticks, len(x_ticks)//2)  # Remove the middle tick
            x_ticks_arcsec = np.delete(x_ticks_arcsec, len(x_ticks_arcsec)//2)  # Remove the middle value
            
            # Do the same for y-axis
            y_ticks = np.delete(y_ticks, len(y_ticks)//2)  # Remove the middle tick
            y_ticks_arcsec = np.delete(y_ticks_arcsec, len(y_ticks_arcsec)//2)  # Remove the middle value
            
            # Manually add 0 for the center tick, ensuring it is labeled as zero
            x_ticks_arcsec = np.concatenate(([0], x_ticks_arcsec))
            x_ticks = np.concatenate(([centre], x_ticks))
            
            y_ticks_arcsec = np.concatenate(([0], y_ticks_arcsec))
            y_ticks = np.concatenate(([centre], y_ticks))
            
            # Apply the new ticks to the plot, rounding for readability
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            
            plt.title("Original Image")
            plt.savefig(os.path.join(save_folder, 'original_image.png'), bbox_inches='tight')
            plt.clf()  # Clear the figure after saving
        
            # Save the plot with the points
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.errorbar(x_coords_final, y_coords_final, xerr=final_err_lst_x, yerr=final_err_lst_y, fmt='o', color = 'red', markersize=1, capsize=0, elinewidth=0.3, label='Binned Data Points')
            #plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1, label="Raw")
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Image with data points")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'image_with_points.png'), bbox_inches='tight')
            plt.clf()
        
            # Save the plot with the free ellipse (with points)
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)
            plt.errorbar(x_coords_final, y_coords_final, xerr=final_err_lst_x, yerr=final_err_lst_y, fmt='o', color = 'red', markersize=1, capsize=0, elinewidth=0.3, label='Binned Data Points')
            #plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Free Ellipse w/points")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'free_ellipse_with_points.png'), bbox_inches='tight')
            plt.clf()
            
            if const_fit == True:
                # Save the plot with the constrained ellipse (with points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)
                plt.errorbar(x_coords_final, y_coords_final, xerr=final_err_lst_x, yerr=final_err_lst_y, fmt='o', color = 'red', markersize=1, capsize=0, elinewidth=0.3, label='Binned Data Points')
                #plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Constrained Ellipse w/points")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'constrained_ellipse_with_points.png'), bbox_inches='tight')
                plt.clf()
        
                # Save the plot with both ellipses (with points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4], zorder=3)
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)
                plt.errorbar(x_coords_final, y_coords_final, xerr=final_err_lst_x, yerr=final_err_lst_y, fmt='o', color = 'red', markersize=1, capsize=0, elinewidth=0.3, label='Binned Data Points')
                #plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Both Ellipses w/points")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'both_ellipses_with_points.png'), bbox_inches='tight')
                
                plt.clf()
        
            # Save the plot with the free ellipse (without points)
            plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
            plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)  # Only the free ellipse without points
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Free Ellipse")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'free_ellipse_without_points.png'), bbox_inches='tight')
            plt.clf()
        
            if const_fit == True:
                # Save the plot with the constrained ellipse (without points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)  # Only the constrained ellipse without points
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Constrained Ellipse")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'constrained_ellipse_without_points.png'), bbox_inches='tight')
                plt.clf()
            
                # Save the plot with both ellipses (without points)
                plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
                plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)  # Both ellipses without points
                plt.xlim(min_x-xborder, max_x+xborder)
                plt.ylim(min_y-yborder, max_y+yborder)
                plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
                plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
                plt.xlabel(r'$\Delta$ RA (arcsec)')
                plt.ylabel(r'$\Delta$ DEC (arcsec)')
                plt.title("Both Ellipse Fits")
                plt.legend(fontsize='xx-small')
                plt.savefig(os.path.join(save_folder, 'both_ellipses_without_points.png'), bbox_inches='tight')
                plt.clf()
            
            #Only points and ellipses, no image
            plt.plot(x, y, markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4],zorder=3)
            if const_fit == True:
                plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)  # Both ellipses with points
            plt.plot(x_coords_final, y_coords_final, 'o', color = 'red', markersize=1, label='Binned Data Points')
            plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
            plt.xlim(min_x-xborder, max_x+xborder)
            plt.ylim(min_y-yborder, max_y+yborder)
            plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
            plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ DEC (arcsec)')
            plt.title("Both Ellipse Fits")
            plt.legend(fontsize='xx-small')
            plt.savefig(os.path.join(save_folder, 'both_ellipses_with_points(no_img).png'), bbox_inches='tight')
            
            plt.clf()
            
            # Filenames of the individual images
            if const_fit == True:
                image_filenames = [
                    'original_image.png',
                    'image_with_points.png',
                    'free_ellipse_with_points.png',
                    'constrained_ellipse_with_points.png',
                    'both_ellipses_with_points.png',
                    'free_ellipse_without_points.png',
                    'constrained_ellipse_without_points.png',
                    'both_ellipses_without_points.png'
                ]
                
                # Create a figure with subplots (2 rows x 4 columns)
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjust figsize as needed
                
                for ax, img_name in zip(axes.flat, image_filenames):
                    img_path = os.path.join(save_folder, img_name)
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis('off')  # Remove axis and ticks
                    ax.set_title('')  # Just to make sure
                
                # Save the final collage
                combined_path = os.path.join(save_folder, 'ALL.png')
                plt.tight_layout(pad=0.5)
                plt.savefig(combined_path, bbox_inches='tight')
                plt.close()
                
            else:
                image_filenames = [
                    'original_image.png',
                    'image_with_points.png',
                    'free_ellipse_with_points.png',
                    'free_ellipse_without_points.png',
                ]
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # Square aspect, adjust if needed
                
                for ax, img_name in zip(axes.flat, image_filenames):
                    img_path = os.path.join(save_folder, img_name)
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis('off')
                
                combined_path = os.path.join(save_folder, 'ALL.png')
                plt.subplots_adjust(wspace=0, hspace=0)  # No space between subplots
                plt.savefig(combined_path, bbox_inches='tight', pad_inches=0)
                plt.close()
# %%

    """ Final Image Displayed ------------------------------- """
        
    final_err_lst_x = np.array(final_err_lst_x)
    final_err_lst_y = np.array(final_err_lst_y)
    plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
    plt.plot(x, y, color='blue', markersize=0.05, label="Free", linestyle='-.', dashes=[4, 2, 1, 4], zorder=3)
    if const_fit == True:
        plt.plot(x_rotated, y_rotated, color='magenta', markersize=0.05, label="Constrained", linestyle='-.', zorder=3)
    plt.errorbar(
        x_coords_final,
        y_coords_final,
        xerr=final_err_lst_x,
        yerr=final_err_lst_y,
        fmt='o', color = 'red',
        markersize=1,
        capsize=0,
        elinewidth=0.3,
        label='Binned data with errors'
    )
    plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1, label="Raw")

    #plt.plot(x_coords2_final, y_coords2_final, 'b,', markersize=0.1)
    plt.xlim(min_x-xborder, max_x+xborder)
    plt.ylim(min_y-yborder, max_y+yborder)
    # Generate tick positions based on min and max values for both axes
    x_ticks = np.linspace(min_x - xborder, max_x + xborder, 7)  # You can adjust number of ticks here
    y_ticks = np.linspace(min_y - yborder, max_y + yborder, 7)
    
    # Convert pixel positions to arcseconds, with center pixel (512, 512) as zero
    x_ticks_arcsec = (x_ticks - centre) * pixel_scale_arcsec
    y_ticks_arcsec = (y_ticks - centre) * pixel_scale_arcsec
    
    # Remove the middle tick and insert zero
    x_ticks = np.delete(x_ticks, len(x_ticks)//2)  # Remove the middle tick
    x_ticks_arcsec = np.delete(x_ticks_arcsec, len(x_ticks_arcsec)//2)  # Remove the middle value
    
    # Do the same for y-axis
    y_ticks = np.delete(y_ticks, len(y_ticks)//2)  # Remove the middle tick
    y_ticks_arcsec = np.delete(y_ticks_arcsec, len(y_ticks_arcsec)//2)  # Remove the middle value
    
    # Manually add 0 for the center tick, ensuring it is labeled as zero
    x_ticks_arcsec = np.concatenate(([0], x_ticks_arcsec))
    x_ticks = np.concatenate(([centre], x_ticks))
    
    y_ticks_arcsec = np.concatenate(([0], y_ticks_arcsec))
    y_ticks = np.concatenate(([centre], y_ticks))
    
    plt.xlabel(r'$\Delta$ RA (arcsec)')
    plt.ylabel(r'$\Delta$ DEC (arcsec)')
    
    # Apply the new ticks to the plot, rounding for readability
    plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
    plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
    plt.title("Both Ellipses w/points")
    plt.legend(fontsize='xx-small')
    plt.show()
    
    
""" JSON COMBINATION """
    
if json_save == True:
    
    def ellipse_points1(a, b, cx, cy, theta, n_points=200):
        t = np.linspace(0, 2 * np.pi, n_points)
        x = a * np.cos(t)
        y = b * np.sin(t)
        # Rotation matrix
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        return x_rot + cx, y_rot + cy
    
    with open('inner.json') as f:
        data1 = json.load(f)

    #with open('middle.json') as f:
        #data2 = json.load(f)
        
    with open('outer.json') as f:
        data3 = json.load(f)

    # Extract ellipses from both files
    ellipses1 = [
        (data1, 'inner - free', 'ellipse_free', 'blue'),
        #(data1, 'inner - const', 'ellipse_const', 'magenta'),
        #(data2, 'middle - free', 'ellipse_free', 'blue'),
        #(data2, 'middle - const', 'ellipse_const', 'magenta'),
        (data3, 'outer - free', 'ellipse_free', 'blue'),
        #(data3, 'outer - const', 'ellipse_const', 'magenta')
    ]
    
    name_part = filename.split('_SPHERE')[0]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(image_data_test_out[id_no], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
    for d, label, key_type, color in ellipses1:
        key = list(d.keys())[0]
        params = d[key][key_type]
        x_json, y_json = ellipse_points1(params["a"], params["b"], params["cx"], params["cy"], params["theta"])
        plt.plot(x_json, y_json, linestyle='-.', dashes=[4, 2, 1, 4], color=color, linewidth=2)
        
    plt.xlim(min_x-xborder, max_x+xborder)
    plt.ylim(min_y-yborder, max_y+yborder)
    plt.xticks(x_ticks, np.round(x_ticks_arcsec, 2))
    plt.yticks(y_ticks, np.round(y_ticks_arcsec, 2))
    plt.xlabel(r'$\Delta$ RA (arcsec)')
    plt.ylabel(r'$\Delta$ DEC (arcsec)')
    plt.title(f"{name_part}")
    plt.savefig('Combined.png', bbox_inches='tight')

            

            
