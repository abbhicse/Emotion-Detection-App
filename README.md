# Emotion Detection from Facial Images

A deep learning-based project that allows users to upload a facial image and detect the expressed emotion using advanced Convolutional Neural Networks. The application is built with **Streamlit** and provides support for multiple deep learning models including **Custom CNN**, **ResNet50**, and **ConvNeXt V2 Tiny**.

---

## Project Highlights

- **Face Detection**: Automatically detects and aligns faces using MediaPipe or Dlib.
- **Emotion Classification**: Classifies emotions into 7 categories using trained CNN models.
- **Model Switching**: Users can switch between three trained models in the web interface.
- **Real-Time Inference**: Results displayed instantly after uploading an image.
- **Landmark Extraction**: Optional facial landmark extraction for analysis or debugging.

---

## Emotion Classes

This project supports classification into the following 7 facial expressions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Required packages include:
> `streamlit`, `torch`, `torchvision`, `timm`, `opencv-python`, `Pillow`, `mediapipe`, `matplotlib`, `dlib` (optional)

---

## Run the Streamlit Application

```bash
streamlit run app.py
```

- Upload a **clear facial image** in `.jpg` or `.png` format.
- Select one of the available models from the dropdown.
- View the predicted emotion directly on the page.

---

## Training Your Own Models

Each of the model training scripts can be used independently:

- **Custom CNN**:

```bash
  python model_cnn.py
  ```

- **ResNet50**:

  ```bash
  python model_resnet.py
  ```

- **ConvNeXt V2 Tiny**:

  ```bash
  python model_convnext.py
  ```

These scripts:

- Load data from the `data/train` and `data/test` folders.
- Apply preprocessing, augmentations, and normalization.
- Train the model and save the final weights to the `models/` directory.
- Generate loss curves in the `plots/` directory.

---

## Dataset

This project uses the [FER-2013 Dataset](https://www.kaggle.com/datasets/damnithurts/fer2013-dataset-images), which contains labeled facial expressions in 48x48 grayscale format.

## Performance Evaluation

Each model is evaluated on:

- Accuracy
- Validation loss
- Training loss
- Inference speed

Visualizations of loss curves are saved in the `plots/` folder after training.

---

## Features & Techniques

### Face Detection

- Uses **Dlib** for grayscale images
- Uses **MediaPipe** for RGB images
- Includes automatic resizing and face alignment

### Feature Extraction

- Optional **facial landmark** extraction using MediaPipe
- Useful for advanced tasks like emotion intensity or facial analysis

### Emotion Prediction

- Models are modular and wrapped inside `utils/` for easy integration
- Compatible with `torchscript` and further deployment

---

## Ethical Considerations

- **Bias Awareness**: FER-2013 may have biases across age, gender, and ethnicity.
- **Privacy**: This app does not store uploaded images.
- **Usage Disclaimer**: Not intended for clinical or real-world decision-making applications.

---

## References

- [FER-2013 Dataset](https://www.kaggle.com/datasets/damnithurts/fer2013-dataset-images)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [ConvNeXt - Facebook AI](https://arxiv.org/abs/2201.03545)
- [Dlib](http://dlib.net/)
- [MediaPipe](https://google.github.io/mediapipe/)

---

## Author

**Your Name**
Project for [Your Institution / Course Name]

---
