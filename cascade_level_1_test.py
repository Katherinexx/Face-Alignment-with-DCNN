import tensorflow as tf
import numpy as np
import pickle
import utils
import cv2
########################################################
###############Loading Test Data########################
########################################################



test_files = []
test_files.append('./data/CNN_data/my_version_helen_test_128_128_color_mirror_CNN.pkl')
test_files.append('./data/CNN_data/my_version_lfpw_test_128_128_color_mirror_CNN.pkl')
test_data = []
for test_file in test_files:
	test_data.extend(pickle.load(open(test_file,'rb')))
test_images=[]
test_labels=[]
test_images[:] = [data.image for data in test_data]
test_labels[:] = [data.landmarks for data in test_data]

test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)
test_labels = np.reshape(test_labels,(test_labels.shape[0],2*test_labels.shape[1]))
test_size = test_images.shape[0]

########################################################
#############Given a non face image#####################
########################################################
# img=cv2.imread("/home/yongzhe/Downloads/header.jpg")
# img=cv2.resize(img,(128,128))
# img = (img-np.mean(img))/np.std(img)
# print(img)
# test_images=[]
# test_images.append(img)
# test_images = np.asarray(test_images)
# b=np.concatenate((test_images,test_images),axis=0)
# for i in range(1106):
# 	b=np.concatenate((b,test_images),axis=0)
# model_path = './model/local_minimum.ckpt'
# saver = tf.train.import_meta_graph(model_path+'.meta')
# with tf.Session() as sess:
#     saver.restore(sess, model_path)
#     # all_vars = tf.trainable_variables()
#     test_prediction = tf.get_collection('test_prediction')[0]
#     test_images_node = tf.get_collection('test_images_node')[0]
#     prediction = sess.run(test_prediction,
#     				feed_dict={test_images_node: b[:, ...]})
# print(prediction)
# prediction = np.reshape(prediction,(prediction.shape[0],68,2))
# for i, new_my_image in enumerate(test_images):
# 	# # Show images and landmarks
# 	image_shown = utils.add_landmarks(new_my_image, prediction[i])
# 	cv2.imshow('image',image_shown)
# 	cv2.waitKey(0) & 0xFF


#########################################################
#########################################################
model_path = './model/mean_shape_centered.ckpt'
saver = tf.train.import_meta_graph(model_path+'.meta')
with tf.Session() as sess:
    saver.restore(sess, model_path)
    # all_vars = tf.trainable_variables()
    test_prediction = tf.get_collection('test_prediction')[0]
    test_images_node = tf.get_collection('test_images_node')[0]
    prediction = sess.run(test_prediction,
    				feed_dict={test_images_node: test_images[:, ...]})


##########################################################

print(prediction)
prediction = np.reshape(prediction,(prediction.shape[0],68,2))
test_labels = np.reshape(test_labels,(test_labels.shape[0],68,2))
for i, new_my_image in enumerate(test_images):
	# # Show images and landmarks
	image_shown = utils.add_landmarks(new_my_image, prediction[i])
	cv2.imshow('image',cv2.cvtColor(image_shown, cv2.COLOR_RGB2BGR))
	cv2.waitKey(0) & 0xFF
