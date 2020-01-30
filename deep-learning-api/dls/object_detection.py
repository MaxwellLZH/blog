from fastapi import UploadFile
import requests
from io import BytesIO
import tensorflow as tf
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from pathlib import Path
from typing import Union, Tuple, List, Dict, Any, IO
from tempfile import SpooledTemporaryFile
import json



def load_category_mapping() -> Dict[int, str]:
    """ Return a mapping from category id to its display name, ex 1 -> person """
    with open('./data/coco_label_mapping.json', 'r') as f:
        mapping = json.load(f)
    return {i['id']: i['display_name'] for i in mapping}


def read_image_from_url(url: str) -> Image:
    return Image.open(BytesIO(requests.get(url).content))


def preprocess_image(path: str) -> Image:
    """ Read a image from local file or URL and do some preprocessing """
    if isinstance(path, SpooledTemporaryFile):
        # from UploadFile.file
        img = Image.open(path)
    elif path.startswith('http'):
        img = read_image_from_url(path)
    
    # minimum preprocessing
    img = img.convert('RGB')
    # img = img.resize((256, 256), resample=Image.ANTIALIAS)
    return img


###############################################################
## Functions adapted from Tensorflow object detection API  ####
###############################################################

def draw_bounding_box_on_image(image: Image,
                               ymin: int,
                               xmin: int,
                               ymax: int,
                               xmax: int,
                               color: Union[Tuple[int], str],
                               font: str,
                               thickness :int = 4,
                               display_str_list: Tuple[str] = ()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image: Image, 
               boxes: List[Tuple[int]], 
               class_names: List[str], 
               scores: List[float], 
               category_mapping: Dict[int, str],
               max_boxes: int = 10, 
               min_score: float = 0.1) -> Image:
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())
  image = np.array(image)

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(category_mapping[int(class_names[i])], int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return Image.fromarray(image)


def run_inference_for_single_image(model, image: Image) -> Dict[str, Any]:
    input_tensor = tf.convert_to_tensor(np.asarray(image))[tf.newaxis,...]

    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    return output_dict


def load_model():
    # for a list of other pretrained models, check out: 
    # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    model = tf.saved_model.load('./models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/saved_model/')
    model = model.signatures['serving_default']
    return model



model, category_mapping = None, None


def make_prediction(f: UploadFile, min_score: float = 0.2) -> Image:
    global model, category_mapping
    if model is None:
        model, category_mapping = load_model(), load_category_mapping()
    

    img = preprocess_image(f.file)
    inference_result = run_inference_for_single_image(model, img)
    drawed = draw_boxes(image=img, 
                        boxes=inference_result['detection_boxes'], 
                        class_names=inference_result['detection_classes'], 
                        scores=inference_result['detection_scores'],
                        category_mapping=category_mapping,
                        min_score=min_score)
    return drawed
