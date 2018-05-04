<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# Installation

    virtualenv --no-site-packages p2tf1.4
    pip install --upgrade tensorflow-gpu==1.7  
    
>functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
AttributeError: 'module' object has no attribute 'data'

#COCO API 安装  

``` 
    sudo pip install Cython  
    sudo pip install fasttext
```

- git clone https://github.com/cocodataset/cocoapi.git
- cd cocoapi / PythonAPI && mv ../common ./
- / *更新所有拥有../common引用的文件 - 用“common”替换“../common”* /
- / *将“REQUIRED_PACKAGES = ['Cython> = 0.28.1']”添加到setup.py * /
- cd .. && tar -czf pycocotools-2.0.tar.gz PythonAPI /
- make

Protobuf Compilation  
https://github.com/tensorflow/models/issues/1834  

    cd tensorflow/models/research/
    /home/wangjinchao/tensorflow/protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=.  

    /home/wangjinchao/tensorflow/protoc_3.5.0/bin/protoc object_detection/protos/*.proto --python_out=.

    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

    python object_detection/builders/model_builder_test.py
       

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  

    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
    tar zxvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz

## 训练：  

    python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=${定义的Config} \
        --train_dir=${训练结果要存放的目录}  

    python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path="/media/wangjinchao/bankcard/ssd_mobilenet_v1_coco.config" \
        --train_dir="/media/wangjinchao/bankcard/training/"  
    
    python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path="/media/wangjinchao/bankcard/ssd_mobilenet_v2_coco.config" \
        --train_dir="/media/wangjinchao/bankcard/training_v2/"    

---
## 评估：
    python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path="/media/wangjinchao/bankcard/ssd_mobilenet_v2_coco.config" \
    --checkpoint_dir="/media/wangjinchao/bankcard/training_v2/" \
    --eval_dir="/media/wangjinchao/bankcard/Evaluation_v2/"



 python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path="/media/wangjinchao/bankcard/ssd_mobilenet_v1_coco.config" \
    --checkpoint_dir="/media/wangjinchao/bankcard/training/" \
    --eval_dir="/media/wangjinchao/bankcard/Evaluation_v1/"



## TensorBoard监控：  

- localhost:6006  

- tensorboard --logdir="/media/wangjinchao/bankcard/training_v2/"

- tensorboard --logdir= . --port = 6007

## Freeze Model模型导出：
`object_detection_tutorial.ipynb`  

tensorflow/python/tools/freeze_graph.py

https://www.ctolib.com/topics-125559.html  

    python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path "/media/wangjinchao/bankcard/ssd_mobilenet_v1_coco.config" \
    --trained_checkpoint_prefix "/media/wangjinchao/bankcard/training/model.ckpt-9961" \
    --output_directory "/media/wangjinchao/bankcard/result/output_inference_graph.pb"  

trained_checkpoint_prefix应该指定多少代的模型，--trained_checkpoint_prefix /media/wangjinchao/bankcard/training/model.ckpt-9961


    python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path "/media/wangjinchao/bankcard/ssd_mobilenet_v2_coco.config" \
    --trained_checkpoint_prefix "/media/wangjinchao/bankcard/training_v2/model.ckpt-5000" \
    --output_directory "/media/wangjinchao/bankcard/result2/output_inference_graph.pb"  