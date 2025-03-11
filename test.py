import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import dlib

# Emotion labels mapping
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

def load_test_image(image_path, target_size=(48, 48)):
    """
    Load and preprocess a test image
    
    Parameters:
    -----------
    image_path : str
        Path to the test image file
    target_size : tuple
        Size to resize the image to (height, width)
        
    Returns:
    --------
    tuple
        Original image, processed image ready for prediction
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize face detector
    detector = dlib.get_frontal_face_detector()
    
    # Detect faces
    faces = detector(gray)
    
    if len(faces) == 0:
        raise ValueError(f"No faces detected in {image_path}")
    
    if len(faces) > 1:
        print(f"Warning: Multiple faces detected in {image_path}. Using the first one.")
    
    # Extract face ROI
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to model input size
    resized_face = cv2.resize(face_roi, target_size)
    
    # Normalize pixel values
    normalized_face = resized_face / 255.0
    
    # Reshape for model input
    input_face = normalized_face.reshape(1, target_size[0], target_size[1], 1)
    
    # Draw rectangle on original image for visualization
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img_rgb, input_face

def test_model_on_image(model_path, image_path, output_dir):
    """
    Test the model on a single image
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model file
    image_path : str
        Path to the test image file
    output_dir : str
        Directory to save the results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the model
    model = keras.models.load_model(model_path)
    
    # Load and preprocess image
    img_rgb, input_face = load_test_image(image_path)
    
    # Make prediction
    prediction = model.predict(input_face)[0]
    emotion_id = np.argmax(prediction)
    confidence = prediction[emotion_id] * 100
    
    # Get emotion label
    emotion = emotion_labels[emotion_id]
    
    # Display and save results
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f"Predicted Emotion: {emotion} ({confidence:.2f}%)")
    plt.axis('off')
    
    output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
    plt.savefig(output_path)
    plt.close()
    
    print(f"Prediction results:")
    print(f"Emotion: {emotion}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Result saved to {output_path}")
    
    return emotion, confidence

def test_model_on_directory(model_path, test_dir, output_dir):
    """
    Test the model on all images in a directory
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model file
    test_dir : str
        Directory containing test images
    output_dir : str
        Directory to save the results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the model
    model = keras.models.load_model(model_path)
    
    # Get list of image files
    image_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {test_dir}")
        return
    
    # Process each image
    results = []
    for img_path in image_files:
        try:
            print(f"Processing {img_path}...")
            img_rgb, input_face = load_test_image(img_path)
            
            # Make prediction
            prediction = model.predict(input_face)[0]
            emotion_id = np.argmax(prediction)
            confidence = prediction[emotion_id] * 100
            
            # Get emotion label
            emotion = emotion_labels[emotion_id]
            
            # Save the result
            plt.figure(figsize=(10, 8))
            plt.imshow(img_rgb)
            plt.title(f"Predicted: {emotion} ({confidence:.2f}%)")
            plt.axis('off')
            
            output_filename = f"result_{os.path.basename(img_path)}"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path)
            plt.close()
            
            # Store result
            results.append({
                'image': os.path.basename(img_path),
                'emotion': emotion,
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Generate summary report
    if results:
        summary_path = os.path.join(output_dir, 'summary_report.txt')
        with open(summary_path, 'w') as f:
            f.write("Emotion Recognition Summary Report\n")
            f.write("=================================\n\n")
            
            for result in results:
                f.write(f"Image: {result['image']}\n")
                f.write(f"Predicted Emotion: {result['emotion']}\n")
                f.write(f"Confidence: {result['confidence']:.2f}%\n\n")
        
        print(f"Summary report saved to {summary_path}")
    
    return results

def test_model_on_video(model_path, video_path, output_dir):
    """
    Test the model on a video file
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model file
    video_path : str
        Path to the video file
    output_dir : str
        Directory to save the results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the model
    model = keras.models.load_model(model_path)
    
    # Initialize face detector
    detector = dlib.get_frontal_face_detector()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter object
    output_path = os.path.join(output_dir, f"result_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         fps, 
                         (frame_width, frame_height))
    
    frame_count = 0
    emotion_counts = {emotion: 0 for emotion in emotion_labels.values()}
    
    print(f"Processing video {video_path}...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 3rd frame to improve performance
        if frame_count % 3 != 0:
            out.write(frame)
            continue
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        if len(faces) == 0:
            # No face detected
            cv2.putText(frame, "No face detected", (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif len(faces) > 1:
            # Multiple faces detected
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            cv2.putText(frame, "Multiple faces detected", (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Single face detected
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess face
            try:
                # Resize to model input size
                resized_face = cv2.resize(face_roi, (48, 48))
                
                # Normalize pixel values
                normalized_face = resized_face / 255.0
                
                # Reshape for model input
                input_face = normalized_face.reshape(1, 48, 48, 1)
                
                # Make prediction
                prediction = model.predict(input_face)[0]
                emotion_id = np.argmax(prediction)
                confidence = prediction[emotion_id] * 100
                
                # Get emotion label
                emotion = emotion_labels[emotion_id]
                
                # Update emotion counts
                emotion_counts[emotion] += 1
                
                # Draw rectangle and emotion text
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display emotion and confidence
                text = f"{emotion}: {confidence:.2f}%"
                cv2.putText(frame, text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Error processing face: {e}")
        
        # Write frame to output video
        out.write(frame)
        
        # Display progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Release video objects
    cap.release()
    out.release()
    
    # Generate summary report
    summary_path = os.path.join(output_dir, 'video_analysis_report.txt')
    with open(summary_path, 'w') as f:
        f.write("Video Emotion Analysis Report\n")
        f.write("============================\n\n")
        f.write(f"Video: {os.path.basename(video_path)}\n")
        f.write(f"Total frames processed: {frame_count}\n\n")
        f.write("Emotion Distribution:\n")
        
        total_emotions = sum(emotion_counts.values())
        if total_emotions > 0:
            for emotion, count in emotion_counts.items():
                percentage = (count / total_emotions) * 100
                f.write(f"{emotion}: {count} frames ({percentage:.2f}%)\n")
        else:
            f.write("No emotions detected in the video.\n")
    
    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Analysis report saved to {summary_path}")
    
    return emotion_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test emotion classification model')
    parser.add_argument('--model_path', type=str, default='./model_output/emotion_classifier.keras',
                        help='Path to the trained model file')
    parser.add_argument('--mode', type=str, choices=['image', 'directory', 'video'], default='image',
                        help='Test mode: single image, directory of images, or video')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input image, directory, or video file')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save test results')
    
    args = parser.parse_args()
    
    print(f"Testing model in {args.mode} mode")
    print(f"Model path: {args.model_path}")
    print(f"Input path: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    
    if args.mode == 'image':
        test_model_on_image(args.model_path, args.input_path, args.output_dir)
    elif args.mode == 'directory':
        test_model_on_directory(args.model_path, args.input_path, args.output_dir)
    elif args.mode == 'video':
        test_model_on_video(args.model_path, args.input_path, args.output_dir)