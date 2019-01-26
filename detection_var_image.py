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


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


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

TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4)]


# Size, in inches, of the output images,use to plt.figure(figsize=IMAGE_SIZE)
# IMAGE_SIZE = (12, 8)

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

def image_preprocessing(img):
  # image_gray = img
  image_gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
  #  image_gray = cv2.medianBlur(image_gray, 3)
  #  image_gray = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
  #  adaptiveThreshold not good ,just try it.
  #  image_gray = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  
  return image_gray
# box_to_color_map{(xmin,xmax,ymin,ymax)(***): 'color'}
# box_to_display_str_map{(xmin,xmax,ymin,ymax)(don't no): ['label: xx%']}
def img_ocr(image_name, output_path, image_org, box_to_color_map, box_to_display_str_map, lang = 'cont41'):
  cont_num_find = 0
  img_label = []
  # Convert coordinates to raw pixels.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box  
    # loads the original image, visualize_boxes_and_labels_on_image_array returned image had draw bounding boxs on it.
    image_corp_org = Image.fromarray(np.uint8(image_org))
    img_width, img_height = image_corp_org.size
    new_xmin = int(xmin * img_width)
    new_xmax = int(xmax * img_width)
    new_ymin = int(ymin * img_height)
    new_ymax = int(ymax * img_height)   
    # Increase cropping security boundary(px).
    offset = 5
    if new_xmin - offset >= 0:
      new_xmin = new_xmin - offset
    if new_xmax + offset <= img_width:
      new_xmax = new_xmax + offset
    if new_ymin - offset >= 0:
      new_ymin = new_ymin - offset
    if new_ymax + offset <= img_height:
      new_ymax = new_ymax + offset
    # Get the label name of every bounding box,and rename 'xxx: 90%' to 'xxx-90%'.
    img_label_name = box_to_display_str_map[box][0].split(': ')
    # Corp image. Note that the PLI and Numpy coordinates are reversed!!!
    image_corp_org = load_image_into_numpy_array(image_org)[new_ymin:new_ymax,new_xmin:new_xmax]       
    image_corp_org = Image.fromarray(np.uint8(image_corp_org))   
    # Tesseract OCR
    lang_use = 'eng+'+lang+'+letsgodigital+snum+eng_f'
    if re.match('container_number+', img_label_name[0]):
      cont_num_find += 1
      image_corp_gray = image_preprocessing(image_corp_org)
      if re.match('container_number_v+', img_label_name[0]):
        cont_num = pytesseract.image_to_string(image_corp_gray, lang=lang_use, config='--psm 6')
      elif re.match('container_number_e+', img_label_name[0]):
        cont_num = pytesseract.image_to_string(image_corp_gray, lang=lang_use, config='--psm 6')
      else :
        cont_num = pytesseract.image_to_string(image_corp_gray, lang=lang_use, config='--psm 4')
      # Save corp image to outo_path ,and join lable in name.
      # image_corp_name make up like this :'image_name(input)'_'cont_num_find'_'img_label_name'
      image_corp_name = image_name[:-4]+ '_'+ str(cont_num_find)+ '_'+ img_label_name[0]
      # img_lable[{lable,actual,cont_num,image_corp_name}]
      img_label.append({'lable':img_label_name[0], 'actual':img_label_name[1], 'cont_num':cont_num, 'image_corp_name':image_corp_name})
      image_corp_org.save(os.path.join(output_path) + '/' + image_corp_name + '_org_'+ image_name[-4:])
      cv2.imwrite(os.path.join(output_path) + '/' + image_corp_name + '_gray_'+ image_name[-4:], image_corp_gray)
      file = open(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'cont_num.txt'), 'a')
      file.write(img_label[cont_num_find - 1]['image_corp_name']+ '_' + img_label[cont_num_find - 1]['actual'] + '\n' + img_label[cont_num_find - 1]['cont_num']+ '\n')
      file.close()
  return img_label # image_corp_org, image_corp_gray

def detection():
  image_label =[]
  for image_path in TEST_IMAGE_PATHS:
    image_org = Image.open(image_path, 'r')
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image_org)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image_np_expanded = np.expand_dims(image_np, axis=0)
    image_name = os.path.basename(os.path.join(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
  
    output_path = os.path.join(PATH_TO_TEST_IMAGES_DIR)
  
    # Visualization of the results of a detection.
    image, box_to_color_map, box_to_display_str_map = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
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
    lang = 'cont41'
    img_label = img_ocr(image_name, output_path, image_org, box_to_color_map, box_to_display_str_map, lang)
    # save visualize_boxes_and_labels_on_image_array output image.
    image_name = os.path.basename(os.path.join(image_path))
    output_image_name = image_name[:-4] + '_out' + image_name[-4:]
    image_out = Image.fromarray(image_np)
    image_out.save(os.path.join(PATH_TO_TEST_IMAGES_DIR) + '/'+ output_image_name)
    image_label.append({str(image_name[:-4]): img_label})
  return image_label
  

if __name__ == "__main__":
    print(detection())



  

