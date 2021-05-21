
import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
import dlib

from predictor import FacePointPositionPredictor
#This method is the main class for transforming 2D images to 3D face as it uses the PRN network for generating the semantic features of the facial points
class FaceModelling:

# all the paths and  ariables that are used in the program ahead are intialized in this method
# the resolution of the input and output is set to 256 becasue the image  is further divided by 255    
# the landmark file which is used for generating  the points are also intialzed here 
    def __init__(self, is_dlib_installed = False, prefix = '.'):

        
        self.resolution_input = 256
        self.resolution_output = 256

        
        if is_dlib_installed:
           
            detector_path = os.path.join(prefix, 'Data/net-data/mmod_human_face_detector.dat')
            self.face_detector = dlib.cnn_face_detection_model_v1(
                    detector_path)

        
        self.pos_predictor = FacePointPositionPredictor(self.resolution_input, self.resolution_output)
        image_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(image_path + '.data-00000-of-00001'):
            
            exit()
        self.pos_predictor.restore(image_path)

        
        self.uv_kpt_ind = np.loadtxt(prefix + '/Data/uv-data/uv_kpt_ind.txt').astype(np.int32) 
        self.face_ind = np.loadtxt(prefix + '/Data/uv-data/face_ind.txt').astype(np.int32) 
        self.triangles = np.loadtxt(prefix + '/Data/uv-data/triangles.txt').astype(np.int32)
        
        self.uv_coords = self.generate_uvMap_coords()        

    

    def dlib_detector(self, image):
        return self.face_detector(image, 1)

    def net_forward(self, image):
        
        return self.pos_predictor.predict(image)
#this is the main method of the pRN network whihc read the image as input , then divides them
#a rectangular background is made around each image and then the face is detected
# here the image detected is also divided and made to a 255 block image which is used for further processing        
    def main_process(self, input, image_info = None):
        
        if isinstance(input, str):
            try:
                image = imread(input)
            except IOError:
                print("error opening file: ", input)
                return None
        else:
            image = input

        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])

        if image_info is not None:
            if np.max(image_info.shape) > 4: 
                kpt = image_info
                if kpt.shape[0] > 3:
                    kpt = kpt.T
                left = np.min(kpt[0, :]); right = np.max(kpt[0, :]); 
                top = np.min(kpt[1,:]); bottom = np.max(kpt[1,:])
            else: 
                coverbox = image_info
                left = coverbox [0]; right = coverbox [1]; top = coverbox [2]; bottom = coverbox [3]
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*1.6)
        else:
            detected_faces = self.dlib_detector(image)
            if len(detected_faces) == 0:
                print('no detected face')
                return None

            d = detected_faces[0].rect 
            left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
            size = int(old_size*1.58)

       
        source = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        destination = np.array([[0,0], [0,self.resolution_input - 1], [self.resolution_input - 1, 0]])
        tform = estimate_transform('similarity', source, destination)
        
        image = image/255
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_input, self.resolution_input))


        cropped_pos = self.net_forward(cropped_image)


       
        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2,:].copy()/tform.params[0,0]
        cropped_vertices[2,:] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2,:], z))
        pos = np.reshape(vertices.T, [self.resolution_output, self.resolution_output, 3])
        
        return pos
#this method deals with generated the coordinates for the UV map spaceon which the various tranformation is applied and the resultant is use ahead            
    def generate_uvMap_coords(self):
        resolution = self.resolution_output
        uvMap_coords = np.meshgrid(range(resolution),range(resolution))
        uvMap_coords = np.transpose(np.array(uvMap_coords), [1,2,0])
        uvMap_coords = np.reshape(uvMap_coords, [resolution**2, -1]);
        uvMap_coords = uvMap_coords[self.face_ind, :]
        uvMap_coords = np.hstack((uvMap_coords[:,:2], np.zeros([uvMap_coords.shape[0], 1])))
        return uvMap_coords
    
 #this method is used to get the colours for the box that is detecting the face and then using teh same image for 3D reconstruction   
    def get_colors(self, image, vertices):
        
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1) 
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] 

        return colors
# the landmark points are used for detetcing the points on the faces    
    def get_landmark_points(self, pos):
        
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt


    def get_vertices_images(self, pos):
        
        all_vertices = np.reshape(pos, [self.resolution_output**2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices
# the colour and reshaping of the verticces is done in this method
    def get_colors_texture(self, texture):
        
        all_colors = np.reshape(texture, [self.resolution_output**2, -1])
        colors = all_colors[self.face_ind, :]

        return colors


   








