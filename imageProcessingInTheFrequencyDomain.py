import numpy as np # linear algebra
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import  rgb2gray


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Transfer parameters are Fourier transform spectrogram and filter size
def highPassFiltering(img,size):
    h, w = img.shape[0:2]#Getting image properties
    h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0
    #Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 0
    return img
    
    
#Transfer parameters are Fourier transform spectrogram and filter size
def lowPassFiltering(img,size):
    h, w = img.shape[0:2]#Getting image properties
    h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    img2 = np.zeros((h, w), np.uint8)#Define a blank black image with the same size as the Fourier Transform Transfer
    
    #Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 1, preserving the low frequency part
    img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1
    
    img3=img2*img #A low-pass filter is obtained by multiplying the defined low-pass filter with the incoming Fourier spectrogram one-to-one.
    return img3
    
if __name__ == "__main__":

	image = imread('../input/fouriertransform-image/3.jpeg')
	
	
	#Covert image into greyscale
	grey = rgb2gray(image)
	plt.figure(num=None, figsize=(8,6), dpi=80)
	plt.imshow(grey, cmap='gray')
	
	image_grey_fourier = np.fft.fftshift(np.fft.fft2(grey))
	plt.figure(num=None, figsize=(8,6), dpi=80)
	plt.imshow(np.log(abs(image_grey_fourier)), cmap='gray')	
	
	#High pass filtering 
	shiff_image = highPassFiltering(image_grey_fourier, 3)
	f_size = 15
	fig, ax = plt.subplots(1,3, figsize=(15, 15))
	ax[0].imshow(np.log(abs(image_grey_fourier)), cmap='gray')
	ax[0].set_title('Masked Fourier', fontsize = f_size)
	ax[1].imshow(grey, cmap='gray')
	ax[1].set_title('Greyscale image', fontsize = f_size)
	ax[2].imshow(abs(np.fft.ifft2(shiff_image)), cmap='gray')
	ax[2].set_title('Transformed greyscale image',fontsize = f_size)
	
	#Low pass filtering
	shiff_image = lowPassFiltering(image_grey_fourier, 3)
	f_size = 15
	fig, ax = plt.subplots(1,3, figsize=(15, 15))
	ax[0].imshow(np.log(abs(image_grey_fourier)), cmap='gray')
	ax[0].set_title('Masked Fourier', fontsize = f_size)
	ax[1].imshow(grey, cmap='gray')
	ax[1].set_title('Greyscale image', fontsize = f_size)
	ax[2].imshow(abs(np.fft.ifft2(shiff_image)), cmap='gray')
	ax[2].set_title('Transformed greyscale image',fontsize = f_size)


