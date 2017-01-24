import cv2
import numpy as np
import menpo
import menpo.io as mio
import utils
import os
import pickle

# ########################################################################################################
# # Testing get_pts_for_mirror_image               
# ########################################################################################################
# file_paths=[]
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/afw')
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/helen/trainset')
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/helen/testset')
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/ibug')
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/lfpw/trainset')
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/lfpw/testset')


# for file_path in file_paths:
# 	utils.get_pts_for_mirror_image(file_path)
# 	print('Finished %s' %file_path)
# file_path = './data/ibug_300W_large_face_landmark_dataset/lfpw/testset'
# pkl_path = './data/CNN_data/my_version_lfpw_test_128_128_color_mirror_CNN.pkl'

# utils.get_dataset_pkl_from_file(file_path, pkl_path, crop_scale = 0.1, to_gray_scale = False, resize = True, to_CNN = True)


# for image in mio.import_images('./data/ibug_300W_large_face_landmark_dataset/afw'):
# 	mat=image.as_imageio()
# 	shape=image.landmarks.get('PTS').lms.points
# 	mat = utils.add_landmarks(mat,shape)
# 	cv2.imshow('image',cv2.cvtColor(mat, cv2.COLOR_RGB2BGR))
# 	cv2.waitKey(0) & 0xFF



# ########################################################################################################
# # Testing write_to_pkl             
# ########################################################################################################
# file_paths=[]
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/afw')
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/helen/trainset')
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/helen/testset')
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/ibug')
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/lfpw/trainset')
# file_paths.append('./data/ibug_300W_large_face_landmark_dataset/lfpw/testset')


# pkl_paths= []
# pkl_paths.append('./data/CNN_data/my_version_afw_train_128_128_color_mirror_CNN.pkl')
# pkl_paths.append('./data/CNN_data/my_version_helen_train_128_128_color_mirror_CNN.pkl')
# pkl_paths.append('./data/CNN_data/my_version_helen_test_128_128_color_mirror_CNN.pkl')
# pkl_paths.append('./data/CNN_data/my_version_ibug_train_128_128_color_mirror_CNN.pkl')
# pkl_paths.append('./data/CNN_data/my_version_lfpw_train_128_128_color_mirror_CNN.pkl')
# pkl_paths.append('./data/CNN_data/my_version_lfpw_test_128_128_color_mirror_CNN.pkl')


# for i in range(6):
# 	utils.write_pkl(file_paths[i], pkl_paths[i],to_gray_scale = False, resize = True, to_CNN = True)



# train_files= []
# train_files.append('./data/CNN_data/my_version_afw_train_128_128_color_mirror_CNN.pkl')
# train_files.append('./data/CNN_data/my_version_helen_train_128_128_color_mirror_CNN.pkl')
# train_files.append('./data/CNN_data/my_version_lfpw_train_128_128_color_mirror_CNN.pkl')
# train_files.append('./data/CNN_data/my_version_ibug_train_128_128_color_mirror_CNN.pkl')

# test_files = []
# test_files.append('./data/CNN_data/my_version_helen_test_128_128_color_mirror_CNN.pkl')
# test_files.append('./data/CNN_data/my_version_lfpw_test_128_128_color_mirror_CNN.pkl')

# train_data = []
# test_data = []

# for train_file in train_files:
# 	with open(train_file, 'rb') as handle:
# 		train_data.extend(pickle.load(handle))

# for test_file in test_files:
# 	with open(test_file, 'rb') as handle:
# 		test_data.extend(pickle.load(handle))

# train_images=[]
# test_images=[]
# train_labels=[]
# test_labels=[]

# train_images[:] = [data.image for data in train_data]
# test_images[:] = [data.image for data in test_data]
# train_labels[:] = [data.landmarks for data in train_data]
# test_labels[:] = [data.landmarks for data in test_data]

# # print (train_images[0][64])
# # print (train_labels[0])

# train_images = np.asarray(train_images)
# test_images = np.asarray(test_images)
# train_labels = np.asarray(train_labels)
# test_labels = np.asarray(test_labels)


# mean_label = np.mean(train_labels, axis=0)

# error = np.zeros((1108,68,2))

# for i in range(1108):
# 	error[i] = np.subtract(train_labels[i], mean_label)

# print ((error**2).mean()/2) 

# train_labels = np.reshape(train_labels,(train_labels.shape[0],136))
# test_labels = np.reshape(test_labels,(test_labels.shape[0],136))

# print(train_images.shape)
# print(test_images.shape)
# print(train_labels.shape)
# print(test_labels.shape)



# for new_my_image in train_data:
# 	# # Show images and landmarks
# 	image_shown = utils.add_landmarks(new_my_image.image, new_my_image.landmarks,mean_label)
# 	cv2.imshow('image',cv2.cvtColor(image_shown, cv2.COLOR_RGB2BGR))
# 	cv2.waitKey(0) & 0xFF

