# Installation
Protobuf Compilation  
https://github.com/tensorflow/models/issues/1834  

    cd tensorflow/models/research/
    /home/wangjinchao/tensorflow/protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=.  

    # From tensorflow/models/research/
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

    python object_detection/builders/model_builder_test.py


https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  

    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
    tar zxvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz

训练：  

    python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=${定义的Config} \
        --train_dir=${训练结果要存放的目录}  

    python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path="/media/wangjinchao/bankcard/ssd_mobilenet_v1_coco.config" \
        --train_dir="/media/wangjinchao/bankcard/training/"


functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
AttributeError: 'module' object has no attribute 'data'


    virtualenv --no-site-packages p2tf1.4
    pip install --upgrade tensorflow-gpu==1.7


