import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
import cv2

from api import FaceModelling

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    faceModelling = FaceModelling(is_dlib_installed = True)

  
    input_images = 'InputImages/'
    output_folder = 'InputImages/results'
    types = ('*.png','*.jpg')
    input_images_list= []
    for files in types:
        input_images_list.extend(glob(os.path.join(input_images, files)))
    
    for i, path in enumerate(input_images_list):
        img = imread(path,plugin='matplotlib')
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        foundFaces = faceCascade.detectMultiScale(grayImage,scaleFactor=1.3,minNeighbors=3,minSize=(30, 30))

        for (x, y, w, h) in foundFaces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            colorImage = img[y:y + h, x:x + w]
            imsave(os.path.join(output_folder, str(w) + str(h) + '_faces.jpg'), colorImage)
     
    input_images_list= []
    for files in types:
        input_images_list.extend(glob(os.path.join(output_folder, files)))

    for i, path in enumerate(input_images_list):

        imgName = path.strip().split('/')[-1][:-4]
        # read img
        img = imread(path,plugin='matplotlib')
        [h, w, c] = img.shape
        if c>3:
            img = img[:,:,:3]

        #regress position map
        max_size = max(img.shape[0], img.shape[1])
        if max_size> 1000:
            img = rescale(img, 1000./max_size)
            img = (img*255).astype(np.uint8)
        pos = faceModelling.main_process(img)        
        img = img/255.
        if pos is None:
            continue
        
        verts = faceModelling.get_vertices_images(pos)
        saveVerts = verts.copy()
        saveVerts[:,1] = h - 1 - saveVerts[:,1]

        colorsVerts = faceModelling.get_colors(img, verts)
        writeObjFiles(os.path.join(output_folder, imgName + '.obj'), saveVerts, faceModelling.triangles, colorsVerts)


def writeObjFiles(obj_name, vertices, triangles, colors):
    triangles = triangles.copy()
    triangles += 1
    with open(obj_name, 'w') as f:
        for i in range(vertices.shape[0]):
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)


if __name__ == '__main__':
    main()
