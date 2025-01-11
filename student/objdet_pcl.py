# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import zlib
import open3d as o3d
# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl):
    print("student task ID_S1_EX2")

    # Ensure the point cloud has the correct shape (N, 3)
    if pcl.shape[1] > 3:
        print("Point cloud contains additional columns. Using only x, y, z coordinates.")
        pcl = pcl[:, :3]  # Only keep the first 3 columns (x, y, z)

    # Step 1: Initialize Open3D visualization with a key callback and create a window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="LiDAR Point Cloud Viewer", width=800, height=600)

    # Step 2: Create an instance of the Open3D point-cloud class
    pcd = o3d.geometry.PointCloud()

    # Step 3: Set points in the point-cloud instance
    pcd.points = o3d.utility.Vector3dVector(pcl)

    # Step 4: Add the point-cloud to the visualization
    vis.add_geometry(pcd)

    # Step 5: Define a callback to advance to the next frame with the right arrow key (key code 262)
    def right_arrow_callback(vis):
        print("Right arrow pressed, showing next frame.")
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    # Step 6: Define a callback to close the window with the escape key (key code 256)
    def escape_key_callback(vis):
        print("Escape key pressed, exiting viewer.")
        vis.close()

    # Register the callbacks
    vis.register_key_callback(262, right_arrow_callback)  # Right arrow key
    vis.register_key_callback(256, escape_key_callback)   # Escape key

    # Start the visualization loop
    vis.run()
    vis.destroy_window()

       

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame

    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)

    # step 2 : extract the range and the intensity channel from the range image
    # step 3 : set values <0 to zero
    ri[ri<0]=0.0
    ri_range = ri[:,:,0]
    ri_intensity = ri[:,:,1]

    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range)) #Normalize the Range Channel
    img_range = ri_range.astype(np.uint8)

    # focus on +/- 90 around the image center
    deg90 = int(img_range.shape[1] / 4)
    ri_center = int(img_range.shape[1]/2)
    img_range = img_range[:, ri_center - deg90:ri_center + deg90]
    img_intensity = ri_intensity[:, ri_center - deg90:ri_center + deg90]

    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
   
    # Compute the 1st and 99th percentiles
    intensity_min = np.percentile(img_intensity, 1) # (min reflectivity)
    intensity_max = np.percentile(img_intensity, 99) # (max reflectivity)

    # Clip intensity values to the percentile range
    img_intensity = np.clip(img_intensity, intensity_min, intensity_max)

    # Normalize intensity to an 8-bit scale (0-255)
    img_intensity = 255 * (img_intensity - intensity_min) / (intensity_max - intensity_min)
    img_intensity = img_intensity.astype(np.uint8)


    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    img_range_intensity = np.vstack((img_range, img_intensity)).astype(np.uint8)

    
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    bev_discret_x = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret_x))

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    bev_discret_y = (configs.lim_y[1] - configs.lim_y[0]) / configs.bev_height
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret_y))

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    show_pcl(lidar_pcl_cpy)

    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height, configs.bev_width))

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)

    indices = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_cpy = lidar_pcl_cpy[indices]

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, counts = np.unique(lidar_pcl_cpy[:, :2], axis=0, return_index=True, return_counts=True)
    lidar_top_pcl = lidar_pcl_cpy[counts]

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    intensities = lidar_top_pcl[:, 3]
    intensity_normalized = np.clip(intensities / (np.max(intensities) - np.min(intensities)), 0, 1) * 255
    intensity_map[np.int_(lidar_top_pcl[:, 1]), np.int_(lidar_top_pcl[:, 0])] = intensity_normalized

    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background

    cv2.imshow("BEV Intensity Map", intensity_map.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1)) #not sure 
    
    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    height_map[np.int_(lidar_top_pcl[:, 0]), np.int_(lidar_top_pcl[:, 1])] = lidar_top_pcl[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map

    ## step 3 : temporarily visualize the Height map using OpenCV to make sure that vehicles separate well from the background
    cv2.imshow("BEV Height Map", height_map.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #######
    ####### ID_S2_EX3 END #######       

    # TODO remove after implementing all of the above steps
    lidar_pcl_cpy = []
    lidar_pcl_top = []
    height_map = []
    intensity_map = []

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


