import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Dense, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import argparse

def create_model():
    """
    Creates and returns the CNN model for emotion classification
    """
    model = Sequential()

    # CNN Layer 1
    model.add(Conv2D(64, (3,3), padding='same', input_shape=(48, 48, 1)))  
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # CNN Layer 2
    model.add(Conv2D(128, (5,5), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # CNN Layer 3
    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # CNN Layer 4
    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # CNN Layer 5
    model.add(Conv2D(1024, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # Global Average Pooling instead of Flatten
    model.add(GlobalAveragePooling2D())

    # Dense Layer 1
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Dense Layer 2
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Dense Layer 3
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(7, activation='softmax'))
    
    return model

def train_model(train_dir, test_dir, output_dir, epochs=70, batch_size=32):
    """
    Train the emotion classification model
    
    Parameters:
    -----------
    train_dir : str
        Directory containing training data
    test_dir : str
        Directory containing test data
    output_dir : str
        Directory to save model and results
    epochs : int
        Number of epochs to train
    batch_size : int
        Batch size for training
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Image size
    img_size = (48, 48)
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=0.1
    )
    
    # Only rescaling for test data
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True,
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    # Create the model
    model = create_model()
    
    # Compile the model
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Display model summary
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.001,
        patience=10,
        verbose=1,
        min_delta=0.0001
    )
    
    callbacks = [early_stopping, reduce_lr]
    
    # Train the model
    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Evaluate the model
    print("\nEvaluating the model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_generator.classes
    
    # Classification report
    class_report = classification_report(y_true, y_pred, target_names=list(train_generator.class_indices.keys()))
    print("\nClassification Report:")
    print(class_report)
    
    # Save classification report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write(class_report)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=list(train_generator.class_indices.keys()),
        yticklabels=list(train_generator.class_indices.keys())
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot and save training history
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Save the model
    model_path = os.path.join(output_dir, 'emotion_classifier.keras')
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train emotion classification model')
    parser.add_argument('--train_dir', type=str, default='./dataset/train',
                        help='Directory containing training data')
    parser.add_argument('--test_dir', type=str, default='./dataset/test',
                        help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='./model_output',
                        help='Directory to save model and results')
    parser.add_argument('--epochs', type=int, default=70,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    print("Starting model training with the following parameters:")
    print(f"Training directory: {args.train_dir}")
    print(f"Testing directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    train_model(
        args.train_dir,
        args.test_dir,
        args.output_dir,
        args.epochs,
        args.batch_size
    )