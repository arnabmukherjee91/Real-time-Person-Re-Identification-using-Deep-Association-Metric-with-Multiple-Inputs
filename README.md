
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

- Clone the repository: (https://github.com/arnabmukherjee91/Real-time-Person-Re-Identification-using-Deep-Association-Metric-with-Multiple-Inputs.git) in your project directory.

## 📌 Introduction

Real-time multiple object tracking (MOT) has gained immense relevance in surveillance, autonomous driving, and smart cities. This project implements a lightweight and efficient person detection and tracking pipeline based on **MobileNet-SSD** for object detection and **Deep SORT** for multi-object tracking.

Inspired by the methodology presented in *"Simple Online and Realtime Tracking with a Deep Association Metric"* by Nicolai Wojke, Alex Bewley, Dietrich Paulus, we integrate a fast and lightweight detector with an appearance-based tracker to achieve robust tracking with low computational overhead. While the paper uses MobileNNetSSD v2, our implementation substitutes it with MobileNet-SSD for faster inference on resource-constrained devices.

## 🚀 How to Run

Make sure the following files are present in the working directory:

- `MobileNetSSD_deploy.prototxt`
- `MobileNetSSD_deploy.caffemodel`
- `model_data/mars-small128.pb`

Then run the detection and tracking pipeline:

```bash
python demo.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel
```

Press `Q` or `Esc`  to quit the live feed.

## 🎥 Output

The `output_2.avi` file is a sample video showing the result of the tracking algorithm. Tracked persons are marked with unique IDs.



## 📚 References

1. Nicolai Wojke, Alex Bewley, Dietrich Paulus  
   "Simple Online and Realtime Tracking with a Deep Association Metric"  
   *CVPR 2021.  
   [IEEE Link]((https://arxiv.org/abs/1703.07402)) | DOI: [1703.07402]((https://doi.org/10.48550/arXiv.1703.07402))

> This paper inspired the approach taken in this project. While it uses YOLOv4 for detection, we chose MobileNet-SSD for lighter computation while keeping the Deep SORT tracking framework intact.

## 📄 License

This project is for academic and research purposes. Refer to the respective model and library licenses.
