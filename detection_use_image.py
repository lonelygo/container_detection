#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Kevin Di'


import numpy as np
import os
from skimage import io, data
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf


from collections import defaultdict
import collections
from io import StringIO
import matplotlib as mpl


from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import cv2
import re

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops


from utils import label_map_util

from utils import visualization_utils as vis_util


MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'container_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
    
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'

TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 8) ]


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session(config = tf.ConfigProto(
                    device_count = {"CPU":16},
                    inter_op_parallelism_threads = 5,
                    intra_op_parallelism_threads = 2,
                    )) as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


for image_path in TEST_IMAGE_PATHS:
  image_org = Image.open(image_path, 'r')
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image_org)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  image_name = os.path.basename(os.path.join(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)


  output_path = os.path.join(PATH_TO_TEST_IMAGES_DIR)

  # Visualization of the results of a detection.
  image, box_to_color_map, box_to_display_str_map = vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
    #   image_path,
    #   output_path,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.75,
      line_thickness=2)

  # Crop bounding box to splt images.
 
  # Convert coordinates to raw pixels.
  t = 0
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box  
    # loads the original image, visualize_boxes_and_labels_on_image_array returned image had draw bounding boxs on it.
    image_corp = Image.fromarray(np.uint8(image_org))
    img_width, img_height = image_corp.size
    new_xmin = int(xmin * img_width)
    new_xmax = int(xmax * img_width)
    new_ymin = int(ymin * img_height)
    new_ymax = int(ymax * img_height)
    # Get the label name to the bounding box.
    img_n = box_to_display_str_map[box][0]
    img_name = img_n.replace(': ','-')
    
    # Corp image.Note that the PLI and Numpy coordinates are reversed!!!
    image_corp = load_image_into_numpy_array(image_org)[new_ymin:new_ymax,new_xmin:new_xmax] 
         
    image_corp = Image.fromarray(np.uint8(image_corp))
    
    # Save corp image to outo_path ,and join lable in name.
    if re.match('container_number+', img_name):
      t += 1
    #  image_corp.show()
      image_corp_gray = image_corp
    #  image_corp_gray = cv2.cvtColor(np.asarray(image_corp), cv2.COLOR_BGR2GRAY)
    #  cv2.imwrite((os.path.join(output_path) + '/' + img_name + 'cvtColor' + (str(t)+'_') + os.path.basename(image_path)), image_corp_gray)
    #  cv2.imshow('cv2', image_corp_gray)
    #  image_corp_gray = cv2.medianBlur(image_corp_gray, 3)
    #  cv2.imshow('medianBlur', image_corp_gray)
    #  cv2.imwrite((os.path.join(output_path) + '/' + img_name +'blur' + (str(t)+'_') + os.path.basename(image_path)), image_corp_gray)
    #  image_corp_gray = cv2.threshold(image_corp_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    #  cv2.imshow('theshold', image_corp_gray)
    #  adaptiveThreshold not good ,just try it.
    #  image_corp_gray = cv2.adaptiveThreshold(image_corp_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #  cv2.imshow('Threshold', image_corp_gray)
    #  cv2.imwrite((os.path.join(output_path) + '/tmp/' + image_name[:-4] +'_' + img_name + '_cvtColor_' + str(t) + image_name[-4:]), image_corp_gray)


      file = open(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'cont_num.txt'), 'a')
      if re.match('container_number_v+', img_name):
        cont_num = pytesseract.image_to_string(image_corp_gray, lang='eng+cont41+letsgodigital+eng_f+snum', config='--psm 6')
        file.write('\n' + img_name + '_' + (str(t)+'_') + os.path.basename(image_path) + '\n' + cont_num + '\n')
      elif re.match('container_number_e+', img_name):
        # image_corp_gray = cv2.threshold(image_corp_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
        cont_num = pytesseract.image_to_string(image_corp_gray, lang='eng+cont41+letsgodigital+eng_f+snum', config='--psm 6')
        file.write('\n' + img_name + '_' + (str(t)+'_') + os.path.basename(image_path) + '\n' + cont_num + '\n')
      else :
        cont_num = pytesseract.image_to_string(image_corp_gray, lang='eng+cont41+letsgodigital+eng_f+snum', config='--psm 4')
        file.write('\n' + img_name + '_' + (str(t)+'_') + os.path.basename(image_path) + '\n' + cont_num + '\n')
      # Save corp container number image.
      image_corp.save(os.path.join(output_path) + '/' + image_name[:-4] +'_' + img_name + '_' + str(t) + image_name[-4:])
      file.close()
  
  # save visualize_boxes_and_labels_on_image_array output image.
  image_name = os.path.basename(os.path.join(image_path))
  output_image_name = image_name[:-4] + '_out' + image_name[-4:]
  image_out = Image.fromarray(image_np)
  image_out.save(os.path.join(PATH_TO_TEST_IMAGES_DIR) + '/'+ output_image_name)

#  plt.figure(figsize=IMAGE_SIZE)
#  plt.imshow(image_np)
  
  # save plt output image.
  # plt.savefig(os.path.join(os.path.join(os.path.join(PATH_TO_TEST_IMAGES_DIR)), output_image_name))

#  plt.show()


  

