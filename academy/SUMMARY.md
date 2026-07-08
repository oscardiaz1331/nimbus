# Summary
Finetuned from pretrained default models using a RTX-3060-ti.
| Model | IoU | Dice | Precision | Recall | Accuracy | FPS (RTX-3060-ti) |
|---|---|---|---|---|---|---|
| YOLO11-N-seg | 0.818 | 0.887 | 0.912 | 0.893 | 0.914 | 53.2 |
| YOLO11-S-seg | 0.820 | 0.891 | 0.898 | 0.911 | 0.914 | 48.1 |
| YOLO11-M-seg | 0.819 | 0.889 | 0.926 | 0.881 | 0.918 | 41.2 |
| YOLO26-N-seg | 0.779 | 0.852 | 0.885 | 0.862 | 0.896 | 47.0 |
| YOLO26-S-seg | 0.791 | 0.861 | 0.907 | 0.870 | 0.901 | 47.1 |
| YOLO26-M-seg | 0.775 | 0.849 | 0.882 | 0.855 | 0.893 | 37.0 |
| RF-DETR Nano | 0.827 | 0.895 | 0.892 | 0.923 | 0.919 | 34.1 |
| RF-DETR Small | 0.834 | 0.899 | 0.903 | 0.924 | 0.920 | 33.5 |
| RF-DETR Medium | 0.831 | 0.898 | 0.8904 | 0.920 | 0.921 | 28.9 |