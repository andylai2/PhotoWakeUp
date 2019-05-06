import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import glob
import pdb

img_dir = './images_with_mask'
mask_dir = './masks'

image_arr = np.array(glob.glob(os.path.join(img_dir, '*')))
#mask_arr = np.array(glob.glob(os.path.join(mask_dir, '*')))
output_dir = './images/masked'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

#if not len(image_arr) == len(mask_arr):
#    raise Exception('Mask/Image number mismatch')

N = len(image_arr)
for i in range(N):
    img_path = image_arr[i]
#    mask_path = mask_arr[i]
    img_name = img_path.split('/')[-1]
    mask_name = 'refined_mask_'+img_name
    mask_path = os.path.join(mask_dir, mask_name)
    #img = np.asarray(cv2.imread(img_path))
    img = cv2.imread(img_path)
    b,g,r = cv2.split(img)
    img = np.asarray(cv2.merge([r,g,b]))
#    print(mask_path)
    mask = np.asarray(cv2.imread(mask_path))

    x,y,_ = np.nonzero(mask)

    # create black mask
    black_mask = np.zeros_like(img)
#    print(black_mask.shape,img.shape)
#    print(x)
#    print(y)
#    pdb.set_trace()
    black_mask[x,y,:] = img[x,y,:]
    black_mask_name = 'mask_black_'+img_name
    plt.imsave(os.path.join(output_dir,black_mask_name),black_mask)

    # create white mask
    white_mask = np.ones_like(img)*255
    white_mask[x,y,:] = img[x,y,:]
    white_mask_name = 'mask_white_'+img_name
    plt.imsave(os.path.join(output_dir,white_mask_name),white_mask)

    # create smooth mask
    smooth_mask = gaussian_filter(img,(7,7,1))
    smooth_mask[x,y,:] = img[x,y,:]
    smooth_mask_name = 'mask_smooth_'+img_name
    plt.imsave(os.path.join(output_dir,smooth_mask_name),smooth_mask)
    
    # save orig images in output folder for references
    plt.imsave(os.path.join(output_dir,img_name),img)
