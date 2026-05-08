# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:27:46 2026

@author: Eduardo Patrón Anchondo

Chagas Detection.

This version detect movement in the first 5 frames of the video.
The highlight is fixed to the entiere video.
"""

# Libraries
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

import os

###########################################

# Parameters of Färneback algoritm of the experiment
farneback_params = dict(
    pyr_scale=0.5,  # Image scale (<1) to build pyramids for each image
    levels=3,       # Number of pyramid layers
    winsize=15,     # Averaging window size
    iterations=3,   # Number of iterations at each pyramid level
    poly_n=5,       # Size of the pixel neighborhood for polynomial expansion
    poly_sigma=1.2, # Standard deviation for polynomial expansion
    flags=0         # Additional flags (optional)
    )

def MagnitudeTest(video, cota, frames, x0=0, x1 = 1920):
    # Storage
    pts = np.empty((0,2))
    
    ret, old_frame = video.read()
    if not ret:
        print('No hay video')

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = old_gray[0:1080, x0:x1]

    for _ in range(frames):
        ret, frame = video.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray[0:1080, x0:x1]

        #Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, **farneback_params)
        old_gray = frame_gray

        # Compute magnitude and angle of the flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Coordenates of selected values
        new_pts = np.column_stack(np.where(magnitude > cota))

        # Remove duplications of coordenates
        pts = np.unique(np.vstack((new_pts, pts)), axis = 0)
        
    return pts

##############################################
def clustering(pts, min_pts = 25, radius = 25):
    dbscan = DBSCAN(eps = radius, min_samples = min_pts) 
    labels = dbscan.fit_predict(pts)

    # Cluster selction
    clusters = [k for k in set(labels) if k != -1]

    # Separación de puntos en su respectivo cluster
    ref_cluster = []
    for k in clusters:  # Dividir los puntos iniciales en los clusters
        puntos_en_cluster = [pts[i] for i in range(len(labels)) if labels[i] == k]
        pts_cluster = np.array(puntos_en_cluster)
        mean = int(len(pts_cluster)/2)
        ref_cluster.append(pts_cluster[mean]) # The representative is the half point of the cluster

    return (np.array(ref_cluster))

##############################################
def Highlight(video_name, metric_name):
    
    """Video Capture"""
    video = cv2.VideoCapture(video_name)
    
    """ Low Movement condition """
    metric = Metrics[metric_name]
    
    """ Test Results and Clustering"""
    pts = MagnitudeTest(video, metric[0], 5)
    detect = clustering(pts, min_pts=metric[1])
    
    """ Reset video to the beginning after MagnitudeTest """
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    """ Output folder setup """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "Highlight_v1")
    os.makedirs(output_dir, exist_ok=True)

    """ Output name - extract just the filename, not the full path """
    base_name   = os.path.basename(video_name)          # 'chagas4.MOV'
    base_name   = os.path.splitext(base_name)[0]        # 'chagas4'
    output_name = base_name + '_' + metric_name + '.mov' # 'chagas4_Pos99.mov'
    output_path = os.path.join(output_dir, output_name)
    
    """ Video Writer setup """
    frame_width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = int(video.get(cv2.CAP_PROP_FPS))

    fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
    out          = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    """ Pre-draw mask once with all rectangles """
    mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    for (y, x) in detect:
        top_left     = (int(x) - radius, int(y) - radius)
        bottom_right = (int(x) + radius, int(y) + radius)
        cv2.rectangle(mask, top_left, bottom_right, (0, 255, 0), 2)

    """ Visualization"""
    cv2.namedWindow('Chagas Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Chagas Detection', 800, 500)

    while True:
        ret, frame = video.read()
        if not ret:
            print('End of the capture')
            break
        
        """ Apply mask to frame """
        frame = cv2.addWeighted(frame, 1.0, mask, 1.0, 0)
        
        """ Write frame to output video """
        out.write(frame)

        """ Video Show """
        cv2.imshow('Chagas Detection', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    """ Release resources """
    out.release()
    video.release()
    cv2.destroyAllWindows()
    print(f'Video saved to: {output_path}')
    
#############################################
""" Experiment parameters """
radius = 50

Metrics = {'Pos99' : (2.4856, 25), 'Pos95': (1.8334, 125),
           'InterPos99': (1.7316, 25), 'InterPos95': (1.3905, 125)}

#############################################
video_name = 'Videos/chagas54.MOV'

parameter = 'Pos99'

Highlight(video_name, parameter)
