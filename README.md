
# Code for CVPR2021 paper

# **Zero-shot Instance Segmentation** 

## Code requirements
+ python: python3.7
+ nvidia GPU
+ pytorch1.1.0
+ GCC >=5.4
+ NCCL 2
+ the other python libs in requirement.txt

## Install 

```
conda create -n zsi python=3.7 -y
conda activate zsi

conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch

pip install cython && pip --no-cache-dir install -r requirements.txt
   
python setup.py develop
```

## Dataset prepare


- Download the train and test annotations files for zsi from [annotations](https://drive.google.com/drive/folders/1TLbmDoRiKcMGq1zyVahXtGVTdkvI9Dus?usp=sharing), put all json label file to
    ```
    data/coco/annotations/
    ```

- Download MSCOCO-2014 dataset and unzip the images it to path： 
    ```
    data/coco/train2014/
    data/coco/val2014/
    ```


- **Training**:
     - 48/17 split:
       ```
          chmod +x tools/dist_train.sh
          ./tools/dist_train.sh configs/zsi/train/zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_decoder.py 4
        ```
        
    - 65/15 split:
      ```
      chmod +x tools/dist_train.sh
      ./tools/dist_train.sh configs/zsi/train/zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_65_15_decoder_notanh.py 4
      ```
          
- **Inference & Evaluate**:

    + **ZSI task**:

        - 48/17 split ZSI task:
            - download [48/17](https://drive.google.com/file/d/1MMDcNWHjTTOaPvMYVlypMIXdvdOcSnao/view?usp=sharing) ZSI model, put it in checkpoints/ZSI_48_17.pth
            
            - inference:
                ```
                chmod +x tools/dist_test.sh
                ./tools/dist_test.sh configs/zsi/48_17/test/zsi/zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_decoder.py checkpoints/ZSI_48_17.pth 4 --json_out results/zsi_48_17.json
                ```
            - our results zsi_48_17.bbox.json and zsi_48_17.segm.json can also downloaded from [zsi_48_17_reults](https://drive.google.com/drive/folders/1ZoFP2ihGhgbVdzagC0M9AVUlXAMmYGRO?usp=sharing).
            - evaluate:
                - for zsd performance
                    ```
                    python tools/zsi_coco_eval.py results/zsi_48_17.bbox.json --ann data/coco/annotations/instances_val2014_unseen_48_17.json
                    ```
                - for zsi performance
                    ```
                    python tools/zsi_coco_eval.py results/zsi_48_17.segm.json --ann data/coco/annotations/instances_val2014_unseen_48_17.json --types segm
                    ```
        - 65/15 split ZSI task:
            - download [65/15](https://drive.google.com/file/d/1UZMNQ9a9Gpbn53JGPilzyjl1oiFYyPw5/view?usp=sharing) ZSI model, put it in checkpoints/ZSI_65_15.pth
            
            - inference:
                ```
                chmod +x tools/dist_test.sh
                ./toools/dist_test.sh configs/zsi/65_15/test/zsi/zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_65_15_decoder_notanh.py checkpoints/ZSI_65_15.pth 4 --json_out results/zsi_65_15.json
                ```
            - our results zsi_65_15.bbox.json and zsi_65_15.segm.json can also downloaded from [zsi_65_15_reults](https://drive.google.com/drive/folders/1ZoFP2ihGhgbVdzagC0M9AVUlXAMmYGRO?usp=sharing).
            - evaluate:
                - for zsd performance
                    ```
                    python tools/zsi_coco_eval.py results/zsi_65_15.bbox.json --ann data/coco/annotations/instances_val2014_unseen_65_15.json
                    ```
                - for zsi performance
                    ```
                    python tools/zsi_coco_eval.py results/zsi_65_15.segm.json --ann data/coco/annotations/instances_val2014_unseen_65_15.json --types segm
                    ```

    + **GZSI task**:

        - 48/17 split GZSI task:
            - use the same model file ZSI_48_17.pth in ZSI task   
            - inference:
                ```
                chmod +x tools/dist_test.sh
                ./tools/dist_test.sh configs/zsi/48_17/test/gzsi/zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_decoder_gzsi.py checkpoints/ZSI_48_17.pth 4 --json_out results/gzsi_48_17.json
                ```
            - our results gzsi_48_17.bbox.json and gzsi_48_17.segm.json can also downloaded from [gzsi_48_17_results](https://drive.google.com/drive/folders/1ZoFP2ihGhgbVdzagC0M9AVUlXAMmYGRO?usp=sharing).
            - evaluate:
                - for gzsd
                    ```
                    python tools/gzsi_coco_eval.py results/gzsi_48_17.bbox.json --ann data/coco/annotations/instances_val2014_gzsi_48_17.json --gzsi --num-seen-classes 48
                    ```
                - for gzsi
                    ```
                    python tools/gzsi_coco_eval.py results/gzsi_48_17.segm.json --ann data/coco/annotations/instances_val2014_gzsi_48_17.json --gzsi --num-seen-classes 48 --types segm
                    ```
        - 65/15 split GZSI task:
            - use the same model file ZSI_48_17.pth in ZSI task   
            - inference:
                ```
                chmod +x tools/dist_test.sh
                ./tools/dist_test.sh configs/zsi/65_15/test/gzsi/zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_65_15_decoder_notanh_gzsi.py checkpoints/ZSI_65_15.pth 4 --json_out results/gzsi_65_15.json
                ```
            - our results gzsi_65_15.bbox.json and gzsi_65_15.segm.json can also downloaded from [gzsi_65_15_results](https://drive.google.com/drive/folders/1ZoFP2ihGhgbVdzagC0M9AVUlXAMmYGRO?usp=sharing).
            - evaluate:
                - for gzsd
                    ```
                    python tools/gzsi_coco_eval.py results/gzsi_65_15.bbox.json --ann data/coco/annotations/instances_val2014_gzsi_65_15.json --gzsd --num-seen-classes 65
                    ```
                - for gzsi
                    ```
                    python tools/gzsi_coco_eval.py results/gzsi_65_15.segm.json --ann data/coco/annotations/instances_val2014_gzsi_65_15.json --gzsd --num-seen-classes 65 --types segm
                    ```


# License

ZSI is released under MIT License.


## Citing

If you use ZSI in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@InProceedings{Zheng_2021_CVPR,
    author    = {Zheng, Ye and Wu, Jiahong and Qin, Yongqiang and Zhang, Faen and Cui, Li},
    title     = {Zero-Shot Instance Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2593-2602}
}

```
