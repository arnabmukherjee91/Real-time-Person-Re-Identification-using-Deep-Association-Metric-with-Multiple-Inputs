
# Person Detection and Tracking with MobileNet-SSD and Deep SORT

This project implements real-time person detection and tracking using the MobileNet-SSD model and the Deep SORT tracking algorithm. It uses OpenCV and TensorFlow-based tools for object tracking across video frames.

## 📂 Files

- `demo.py` — Main script for video stream processing, object detection, and tracking.
- `MobileNetSSD_deploy.caffemodel` — Pre-trained Caffe model for person detection.
- `output_2.avi` — Sample output video showing tracking results.
- `model_data/mars-small128.pb` — (Required) Deep SORT feature extractor model.

## 🛠️ Requirements

Make sure you have the following Python packages installed:

```bash
pip install opencv-python imutils numpy pillow
```

Deep SORT dependencies:

- Clone the [Deep SORT repository](https://github.com/nwojke/deep_sort) or include the `deep_sort` and `tools` folders in your project directory.

## 📌 Introduction

Real-time multiple object tracking (MOT) has gained immense relevance in surveillance, autonomous driving, and smart cities. This project implements a lightweight and efficient person detection and tracking pipeline based on **MobileNet-SSD** for object detection and **Deep SORT** for multi-object tracking.

Inspired by the methodology presented in *"Real-Time Multiple Object Tracking Using Deep SORT with YOLOv4"* by Chembilan et al. (2021), we integrate a fast and lightweight detector with an appearance-based tracker to achieve robust tracking with low computational overhead. While the paper uses YOLOv4, our implementation substitutes it with MobileNet-SSD for faster inference on resource-constrained devices.

## 🚀 How to Run

Make sure the following files are present in the working directory:

- `MobileNetSSD_deploy.prototxt`
- `MobileNetSSD_deploy.caffemodel`
- `model_data/mars-small128.pb`

Then run the detection and tracking pipeline:

```bash
python demo.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel
```

Press `Q` to quit the live feed.

## 🎥 Output

The `output_2.avi` file is a sample video showing the result of the tracking algorithm. Tracked persons are marked with unique IDs.

## 🔍 Insights from Reference Paper

- **Detector-Tracker Combination**: The paper combines YOLOv4 with Deep SORT to balance speed and accuracy. In our case, we replace YOLOv4 with MobileNet-SSD for similar real-time performance with a simpler setup.

- **Appearance Features in Tracking**: Deep SORT uses appearance embeddings to associate object identities across frames, significantly improving tracking consistency compared to IoU-based methods alone.

- **Real-Time Performance**: The authors demonstrated real-time processing (30+ FPS) with GPU acceleration, showing the approach is feasible for edge devices and real-world applications.

- **Application Scenarios**: The method has applications in public safety, crowd analysis, and smart surveillance — all relevant to the kind of use cases we target with this implementation.

## 📚 References

1. Chembilan, Vineeth, et al.  
   "Real-Time Multiple Object Tracking Using Deep SORT with YOLOv4."  
   *2021 International Conference on Artificial Intelligence and Smart Systems (ICAIS)*. IEEE, 2021.  
   [IEEE Link](https://ieeexplore.ieee.org/abstract/document/9603073) | DOI: [10.1109/ICAIS50930.2021.9603073](https://doi.org/10.1109/ICAIS50930.2021.9603073)

> This paper inspired the approach taken in this project. While it uses YOLOv4 for detection, we chose MobileNet-SSD for lighter computation while keeping the Deep SORT tracking framework intact.

## 📄 License

This project is for academic and research purposes. Refer to the respective model and library licenses.
