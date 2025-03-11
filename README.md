# Emojify - Real-time Facial Emotion Recognition

Emojify is a Streamlit-based application that recognizes facial emotions in real-time from a webcam feed and displays the corresponding emoji. The application uses a Convolutional Neural Network (CNN) trained on a dataset of facial expressions to classify emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features

- Real-time emotion recognition from webcam feed
- Visualization of detected emotion as an emoji
- Support for single face detection and warning for multiple faces
- Trained CNN model with high accuracy
- Testing functionality for images and videos

## Project Structure

```
emojify/
├── app.py                      # Main Streamlit application
├── train.py                    # Script to train the emotion recognition model
├── test.py                     # Script to test the model on images/videos
├── requirements.txt            # Project dependencies
├── emotion_classifier.keras    # Trained model file
├── emojis/                     # Directory containing emoji images
│   ├── angry.png
│   ├── disgusted.png
│   ├── fearful.png
│   ├── happy.png
│   ├── neutral.png
│   ├── sad.png
│   └── surpriced.png
└── dataset/                    # Dataset directory
    ├── train/                  # Training data
    └── test/                   # Test data
```

## Requirements

- Python 3.8+
- OpenCV
- dlib
- TensorFlow
- Streamlit
- NumPy
- Matplotlib
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emojify.git
cd emojify
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download or prepare the emoji images and place them in the `emojis` folder.

5. For model training, prepare your [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) in the `dataset` directory with the required structure.

## Usage

### Running the Streamlit App

To run the main application:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.

### Training the Model

To train the emotion recognition model:

```bash
python train.py --train_dir ./dataset/train --test_dir ./dataset/test --output_dir ./model_output
```

Additional options:
- `--epochs`: Number of training epochs (default: 70) -mx9
- `--batch_size`: Batch size for training (default: 32)

### Testing the Model

#### Testing on a single image:

```bash
python test.py --mode image --input_path path/to/image.jpg --output_dir ./test_results
```

#### Testing on a directory of images:

```bash
python test.py --mode directory --input_path path/to/images_dir --output_dir ./test_results
```

#### Testing on a video file:

```bash
python test.py --mode video --input_path path/to/video.mp4 --output_dir ./test_results
```

## Model Architecture

The CNN model architecture consists of:
- 5 convolutional layers with batch normalization, ReLU activation, and max pooling
- Global average pooling layer
- 3 dense layers with dropout for regularization
- Final softmax layer for 7-class classification

## Acknowledgements

This project uses a CNN architecture inspired by state-of-the-art models for facial emotion recognition.

## License

This project is licensed under the MIT License - see the LICENSE file for details.