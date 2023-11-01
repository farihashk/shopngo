import cv2
import numpy as np
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import os
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util
import tensorflow as tf
from tkinter import *
from PIL import Image, ImageTk
from data import *

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'

CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'
labels = [{'name': 'Book', 'id': 1}, {'name': 'Cocacola', 'id': 2}, {'name': 'Eraser', 'id': 3},
          {'name': 'Pen', 'id': 4}, {'name': 'Scissors', 'id': 5}]
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

CONFIG_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME + '/pipeline.config'
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = 5
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + \
                                                    '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
    ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + \
                                                      '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
    ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
    f.write(config_text)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(
    model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    initial_detections = detection_model.postprocess(prediction_dict, shapes)
    return initial_detections


category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')


def add_item():
    items = get_item()
    my_listbox.insert(END, items)


def remove_items():
    for item in my_listbox.curselection():
        clear_bag(item - 1)
        my_listbox.delete(item)


def clear_list():
    for item in range(my_listbox.size() - 1):
        clear_bag(item)
    my_listbox.delete(1, END)


def total_amount():
    amount = get_sum()
    l3["text"] = "Total Amount = " + str(amount)


root = Tk()
#root.geometry("1500x600")
root.title("Shop-N-Go")
root.config(bg="black")
Label(root, text="Welcome to Shop-N-Go", font=('Consoles', 25), bg="black", fg="red").grid(row=0, column=0)
f1 = LabelFrame(root, bg="red")
f1.grid(row=1, column=0)
L1 = Label(f1, bg="red")
L1.grid(row=1, column=0)
f2 = Frame(root, bg="black")
f3 = Frame(root, bg="black")
f2.grid(row=0, column=2)
f3.grid(row=3, column=2)
my_listbox = Listbox(root, bg="gray", width=30, height=15, font=('Consoles', 15))
my_listbox.grid(row=1, column=2, sticky= "nsew")
my_listbox.insert(0, "Item        |Price")
Button(f2, text="Add Item To list", bg='black', fg='white', command=add_item).grid(row=0, column=1)
Button(f3, text="Remove From list", bg='black', fg='white', command=remove_items).grid(row=0, column=0)
Button(f3, text="Clear list", bg='black', fg='white', command=clear_list).grid(row=0, column=1)
Button(f3, text="Get Bill", bg='black', fg='white', command=total_amount).grid(row=0, column=2)
l3 = Label(f3, text="Total Ammount", font=('Consoles', 20), bg="black", fg="red")
l3.grid(row=1, column=0)
l2 = Label(f2, text="item", font=('Consoles', 25), bg="black", fg="red")
l2.grid(row=0, column=0, sticky ="nsew")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=2,
        min_score_thresh=0.5,
        agnostic_mode=False)

    # cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    final_img = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
    final_img = ImageTk.PhotoImage(Image.fromarray(final_img))
    L1['image'] = final_img
    root.update()

    score = float((detections['detection_scores'])[0])
    if score > 0.50:
        add_to_list((detections['detection_classes'])[0], l2)

    # print((detections['detection_classes']+label_id_offset)[0])
    # print((detections['detection_scores'])[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

cv2.destroyAllWindows()
