# YoloV3
________
YoloV3 Simplified for training on Colab with custom dataset. 

# We have forged this repository from https://github.com/theschoolofai/YoloV3
# We have train YoloV3 model on our custom dataaset of persons

For custom dataset:
1. We have clone this repo: https://github.com/miki998/YoloV3_Annotation_Tool
2. We have then done the setup installation of project and then Annotated the images using the Annotation tool.
3. We have then placed our Custom annotated images and its labels in this repository inside data folder
```
data
  --customdata
    --images/
      --000000000036.jpg
      --000000000037.jpg
      --...
    --labels/
      --000000000036.txt
      --000000000037.txt
      --...
    custom.data #data file
    custom.names #your class names
    custom.txt #list of name of the images you want your network to be trained on. Currently we are using same file for test/train
```

# After that our run our YoloV3 object detection model to train on our dataset
We used the below command to run our model
Run this command `python train.py --data data/customdata/custom.data --batch 10 --cache --cfg cfg/yolov3-custom.cfg --epochs 10 --nosave`

We have train our model for 10 epochs to get better accuracy

Below is the training logs 

```
Model Summary: 225 layers, 6.25733e+07 parameters, 6.25733e+07 gradients
Image sizes 512 - 512 train, 512 test
Using 8 dataloader workers
Starting training for 10 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
                 all       100       191  0.000302   0.00524   0.00169  0.000571

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
                 all       100       191         0         0   0.00758         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
                 all       100       191         0         0    0.0648         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
                 all       100       191         0         0       0.2         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
                 all       100       191         1   0.00524     0.552    0.0104

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
                 all       100       191         1   0.00524      0.66    0.0104

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
                 all       100       191         1   0.00524     0.745    0.0104

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
                 all       100       191         1   0.00524     0.762    0.0104

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
                 all       100       191         1   0.00524     0.798    0.0104

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
                 all       100       191         1   0.00524     0.776    0.0104
10 epochs completed in 0.809 hours.
```

**Results**
After training for 10 Epochs, results look awesome!

![Trained_images](train_batch0.jpeg)


# Then we have tested our model. We used below command to see our model performance on test

!python test.py --cfg cfg/yolov3-custom.cfg --data data/customdata/custom.data --weights weights/last.pt --batch-size 2 --save-json --task 'test'

test_batch0
![Test_images](test_batch0.jpeg)
