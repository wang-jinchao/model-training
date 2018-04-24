import json
import os
import io
import tensorflow as tf
import PIL.Image
import hashlib

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
  
# dict = {1:'plate_no', 2:'vehicle_type', 3:'owner', 4:'address', 5:'use_character', 6:'model', 7:'vin', 8:'engine_no',
#         9: 'register_date', 10:'issue_date', 11:'issued_by', 12:'card_box'}

dict = {1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: 'number_area'}


f = open('/home/dengjin/RIMA/bankcard/sdk/train.json')
result = json.load(f)

output_path = os.path.join('/home/dengjin/RIMA/bankcard/sdk/tfrecord', 'train.record')
#output_crop_base = 'E:\\PythonExp\\vehicle_license\\data\\252_clean_in_500_crop'
image_base = '/home/dengjin/RIMA/bankcard/sdk/images'
writer = tf.python_io.TFRecordWriter(output_path)

for i in range(len(result)):
    image_file = result[i]['image']
    image_id = image_file[0:image_file.rfind('.')]
    # if int(image_id) <= 480:
    #     continue
    
    for j in range(len(result[i]['labels'])):
        points = result[i]['labels'][j]['points']
        x_min,y_min,x_max,y_max = points[0][0], points[0][1], points[1][0], points[1][1]
    
        # crop image
        img_path = os.path.join(image_base, image_file)
        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        width, height = image.size
    #    cropped = image.crop((x_min * width, y_min * height, x_max * width, y_max * height))
    #    cropped.save(os.path.join(output_crop_base, image_file))
        
        # save tf record
        if image.format != 'JPEG':
            continue
            #raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()
    
        classes = []
        classes_text = []
        poses = []
    
        # for box
    #    classes.append(1)
    #    classes_text.append(dict[12].encode('utf8'))
    #    poses.append('Frontal'.encode('utf8'))
    
        # for item
        category = result[i]['labels'][j]['catygory_id'] + 1
        classes.append(category)
        classes_text.append(dict[category].encode('utf8'))
        poses.append('Frontal'.encode('utf8'))
    
        tf_example = tf.train.Example(features=tf.train.Features(feature={
              'image/height': int64_feature(height),
              'image/width': int64_feature(width),
              'image/filename': bytes_feature(image_file.encode('utf8')),
              'image/source_id': bytes_feature(image_file.encode('utf8')),
              'image/key/sha256': bytes_feature(key.encode('utf8')),
              'image/encoded': bytes_feature(encoded_jpg),
              'image/format': bytes_feature('jpeg'.encode('utf8')),
              'image/object/bbox/xmin': float_list_feature([x_min]),
              'image/object/bbox/xmax': float_list_feature([x_max]),
              'image/object/bbox/ymin': float_list_feature([y_min]),
              'image/object/bbox/ymax': float_list_feature([y_max]),
              'image/object/class/text': bytes_list_feature(classes_text),
              'image/object/class/label': int64_list_feature(classes),
              'image/object/difficult': int64_list_feature(0),
              'image/object/truncated': int64_list_feature(0),
              'image/object/view': bytes_list_feature(poses),
          }))
        
        
        writer.write(tf_example.SerializeToString())
writer.close()



