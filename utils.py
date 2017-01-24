import cv2
import numpy as np
import menpo.io as mio
import os
import pickle

mirror_transfrom = {1:17,2:16,3:15,4:14,5:13,6:12,7:11,8:10,9:9,
					10:8,11:7,12:6,13:5,14:4,15:3,16:2,17:1, # face 
					18:27,19:26,20:25,21:24,22:23,
					23:22,24:21,25:20,26:19,27:18, #eye brow
					37:46,38:45,39:44,40:43,41:48,42:47,
					43:40,44:39,45:38,46:37,47:42,48:41, # eye
					28:28,29:29,30:30,31:31,32:36,33:35,34:34,35:33,36:32, # nose
					49:55,50:54,51:53,52:52,53:51,54:50,55:49,
					61:65,62:64,63:63,64:62,65:61,
					68:66,67:67,66:68,
					60:56,59:57,58:58,57:59,56:60 # lips
					}

def add_landmarks(mat, shape, shape_compare=None):
	if shape.mean()<1:
		shape[:,0] = shape[:,0]*mat.shape[0]
		shape[:,1] = shape[:,1]*mat.shape[1]
	for i in range(0, shape.shape[0]):
		cv2.circle(mat, center=(int(shape[i][1]), int(shape[i][0])), radius=3, color=(0,255,0), thickness=-1)
	if shape_compare is not None:
		if shape_compare.mean()<1:
			shape_compare[:,0] = shape_compare[:,0]*mat.shape[0]
			shape_compare[:,1] = shape_compare[:,1]*mat.shape[1]
			print (shape_compare[:,1])
			print (shape_compare[:,0])
		for i in range(0, shape_compare.shape[0]):
			cv2.circle(mat, center=(int(shape_compare[i][1]), int(shape_compare[i][0])), radius=3, color=(0,0,255), thickness=-1)
	return mat

    
def read_landmarks_from_the_text(txt_path):
	with open(txt_path, 'r') as pts:
		lines = pts.readlines()
	land_marks=[]
	for i in range(3,71):
		p=[]
		p.append(float(lines[i].split()[0]))
		p.append(float(lines[i].split()[1]))
		land_marks.append(p)
	return land_marks

def get_pts_for_mirror_image(file_path, verbose = False):
	for image_file_name in mio.image_paths(file_path):
		if os.path.exists(os.path.splitext(str(image_file_name))[0]+'_mirror.jpg') and not os.path.exists(os.path.splitext(str(image_file_name))[0]+'_mirror.pts'):
			img = mio.import_image(image_file_name).mirror()
			mirrored_points = []
			for i in range(68):
				unmirrored_points = img.landmarks.get('PTS').lms.points
				mirrored_points.append([unmirrored_points[mirror_transfrom[i+1]-1][1], unmirrored_points[mirror_transfrom[i+1]-1][0]])
			if verbose:
				print ('Adding mirror pts for: ' + image_file_name)

			with open(os.path.splitext(str(image_file_name))[0]+'_mirror.pts', 'w') as f:
				f.write('version: 1\nn_points:  %d\n{\n' %(len(mirrored_points)))
				for point in mirrored_points:
					f.write('%f %f\n' %(point[0],point[1]))
				f.write('}')


def write_pkl(file_path, pkl_path,
					crop_scale = 0.1, 
					to_gray_scale = True, 
					resize = False, resize_shape = (128,128), resize_order = 1,
					to_CNN = False):
	if os.path.exists(file_path):
		my_images = []
		print('Digging into %s' % file_path)
	else:
		raise IOError('File path %s not found. Please have it checked' %file_path)
		exit()
	for i,image in enumerate(mio.import_images(file_path, verbose = True)):
		if image.has_landmarks and 'PTS' in image.landmarks.group_labels:
			new_my_image = my_image(image,
										crop_scale = crop_scale, 
										to_gray_scale=to_gray_scale,
										resize=resize,resize_shape = resize_shape, resize_order = resize_order,
										to_CNN=to_CNN)
			my_images.append(new_my_image)
			# # Show images and landmarks
			# image_shown = utils.add_landmarks(new_my_image.image, new_my_image.landmarks)
			# cv2.imshow('image',cv2.cvtColor(image_shown, cv2.COLOR_RGB2BGR))
			# cv2.waitKey(0) & 0xFF
			print ('%d / %d in dataset' % (i, len(mio.import_images(file_path))))
		else:
			print("Landmarks not found!!!")
	print('Finished loading %s' % file_path)
	print('Writing to %s, this may take quite a long time.' % pkl_path)
	mio.export_pickle(my_images, pkl_path)
	print('Finished writing to %s.' % pkl_path)
	# cv2.destroyAllWindows()

def get_average_shape_centered_data():
	# Return:
	# 1. Train image
	# 2. Average shape centered train label
	# 3. Average shape
	train_files= []
	train_files.append('./data/CNN_data/my_version_afw_train_128_128_color_mirror_CNN.pkl')
	train_files.append('./data/CNN_data/my_version_helen_train_128_128_color_mirror_CNN.pkl')
	train_files.append('./data/CNN_data/my_version_lfpw_train_128_128_color_mirror_CNN.pkl')
	train_files.append('./data/CNN_data/my_version_ibug_train_128_128_color_mirror_CNN.pkl')
	train_data = []
	for train_file in train_files:
		train_data.extend(pickle.load(open(train_file,'rb')))
	train_images=[]
	train_labels=[]
	train_images[:] = [data.image for data in train_data]
	train_labels[:] = [data.landmarks for data in train_data]
	# train images shape (6566,128,128,3)
	# train label shape (6566, 136)
	train_images = np.asarray(train_images)
	train_labels = np.asarray(train_labels)
	# Change shape from (n, 68, 2) to (n,136)
	train_labels = np.reshape(train_labels,(train_labels.shape[0],2*train_labels.shape[1]))
	mean_shape = np.mean(train_labels,axis=0)
	centered_train_labels = train_labels - mean_shape

	return train_images, centered_train_labels, mean_shape



class my_image(object):
# Contains two elements

# self.image
# self.landmarks

# Get image coppped
# Get image grayscaled if needed
# Get image resized if needed
# Get normalized if needed
# Get numpy image 
# Get the point coordinates
	def __init__ (self,full_sized_image, 
					crop_scale = 0.1, 
					to_gray_scale = True, 
					resize = False, resize_shape = (128,128), resize_order = 1,
					to_CNN = False):
		img = full_sized_image.crop_to_landmarks_proportion(crop_scale)
		if to_gray_scale:
			img = img.as_greyscale(mode='luminosity')
		if resize:
			img = img.resize(resize_shape,resize_order)
		if to_CNN:
			img = img.normalize_std()
			self.image = img.as_imageio(out_dtype = np.float32)
			tmp_landmarks = img.landmarks.get('PTS').lms.points
			tmp_landmarks[:,0] = tmp_landmarks[:,0]/float(self.image.shape[0])
			tmp_landmarks[:,1] = tmp_landmarks[:,1]/float(self.image.shape[1])
			self.landmarks = tmp_landmarks
		else:
			self.image = img.as_imageio()
			self.landmarks = img.landmarks.get('PTS').lms.points
		if len(self.image.shape) == 2 and not to_gray_scale:
			n_shape = self.image.shape + (3,)
			img_3_ch = np.zeros(n_shape)
			img_3_ch[:,:,0] = self.image
			img_3_ch[:,:,1] = self.image
			img_3_ch[:,:,2] = self.image
			self.image=img_3_ch
