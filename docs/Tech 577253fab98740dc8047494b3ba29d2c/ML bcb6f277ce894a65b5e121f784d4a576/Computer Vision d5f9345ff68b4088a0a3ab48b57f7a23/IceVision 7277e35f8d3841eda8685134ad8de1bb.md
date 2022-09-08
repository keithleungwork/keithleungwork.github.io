# IceVision

---

## Quick Fact :

- [https://airctic.com/0.12.0/install/](https://airctic.com/0.12.0/install/)
- It wraps up fastai, pytorch lightning
- Its unified API allow to choose libraries → their models → their backbones
    
    ```python
    # e.g. lib = mmdet, model = retinanet, backbone = resnet50_fpn_1x
    model_type = models.mmdet.retinanet
    backbone = model_type.backbones.resnet50_fpn_1x(pretrained=True)
    ```
    

## Prerequisite Knowledge :

Basic knowledge of 

- Fastai / Pytorch Lightning
- YOLO

---

## Frequent Issue :

- Issue - Icevision is suck on dependencies, the functional env so far :
    
    ```python
    # packages
    python==3.7.13
    icevision[all]==0.11.0
    yolov5-icevision==6.0.0
    "opencv-python-headless<4.3"
    setuptools==61.2.0
    ```
    

- issue - how to do inference with the exported model ?
    - i.e. how to load trained checkpoints
    - option 1 - official inference method `model_from_checkpoint`
        - weird, need to install mmcv-full even if you use yolo5:
            
            ```python
            pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html
            #mmcv-full-1.5.3.tar.gz
            ```
            
    - option 2 - use fastai / torch to load them manually
    
- issue - training or data loader hanged  locally (CPU)
    - because CPU env lead to deadlock
    - [https://github.com/pytorch/pytorch/issues/1355](https://github.com/pytorch/pytorch/issues/1355)
    - You need to set num of workers to 0

- issue - images / plot not showing
    - you need to put `%matplotlib inline` or [others](../../Python%200a10a99cd01b4ebbba143b2176835b8c/Visualization%20Technique%20a3c73bacb92948debeed202afd709466.md) in the top of notebook
    - [https://github.com/matplotlib/matplotlib/issues/14534](https://github.com/matplotlib/matplotlib/issues/14534)
    
- issue - about “image truncated”, below code can solve it
    
    ```python
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    ```