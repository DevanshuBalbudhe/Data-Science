# =====================================================================
# CELL 1: Install TensorFlow (if needed)
# =====================================================================
# Uncomment below line if TensorFlow is not installed
# !pip install tensorflow

print("Ready to start!")


# =====================================================================
# CELL 2: Import Libraries
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported!")


# =====================================================================
# CELL 3: Import TensorFlow and Keras
# =====================================================================

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    
    print(f"TensorFlow version: {tf.__version__}")
    print("TensorFlow/Keras imported successfully!")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Please install it first.")
    print("Run: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False


# =====================================================================
# CELL 4: Set Random Seeds (for reproducibility)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    np.random.seed(42)
    tf.random.set_seed(42)
    print("Random seeds set for reproducibility")


# =====================================================================
# CELL 5: Load Dataset
# =====================================================================

# Load housing dataset from CSV
df = pd.read_csv('/home/claude/housing_data.csv')

print("Dataset loaded!")
print(f"Shape: {df.shape}")


# =====================================================================
# CELL 6: Display First Few Rows
# =====================================================================

df.head()


# =====================================================================
# CELL 7: Dataset Information
# =====================================================================

df.info()


# =====================================================================
# CELL 8: Statistical Summary
# =====================================================================

df.describe()


# =====================================================================
# CELL 9: Check Missing Values
# =====================================================================

print("Missing values:")
print(df.isnull().sum())


# =====================================================================
# CELL 10: Separate Features and Target
# =====================================================================

# Features (X) - all columns except Price
X = df.drop('Price', axis=1).values

# Target (y) - Price column
y = df['Price'].values

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


# =====================================================================
# CELL 11: Train-Test Split (using sklearn)
# =====================================================================

from sklearn.model_selection import train_test_split

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# =====================================================================
# CELL 12: Feature Scaling (using sklearn)
# =====================================================================

from sklearn.preprocessing import StandardScaler

# Create and fit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaled successfully!")


# =====================================================================
# CELL 13: Build Neural Network Model (using Keras Sequential)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Create Sequential model (layer by layer approach)
    model = models.Sequential()
    
    # Add layers one by one
    # Input layer + First hidden layer (64 neurons, ReLU activation)
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    
    # Second hidden layer (32 neurons, ReLU activation)
    model.add(layers.Dense(32, activation='relu'))
    
    # Third hidden layer (16 neurons, ReLU activation)
    model.add(layers.Dense(16, activation='relu'))
    
    # Output layer (1 neuron, no activation for regression)
    model.add(layers.Dense(1))
    
    print("Model created successfully!")


# =====================================================================
# CELL 14: Display Model Architecture
# =====================================================================

if TENSORFLOW_AVAILABLE:
    model.summary()


# =====================================================================
# CELL 15: Compile Model (set optimizer, loss, metrics)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Compile model
    # optimizer: Adam (adaptive learning rate)
    # loss: MSE (Mean Squared Error for regression)
    # metrics: MAE (Mean Absolute Error)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    print("Model compiled!")


# =====================================================================
# CELL 16: Setup Callbacks (for better training)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Early Stopping: stop training if no improvement
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce Learning Rate: reduce LR when learning plateaus
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
    
    print("Callbacks configured!")


# =====================================================================
# CELL 17: Train Model (using fit method)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70)
    
    # Train the model
    history = model.fit(
        X_train_scaled,           # Training features
        y_train,                  # Training target
        epochs=100,               # Number of epochs
        batch_size=32,            # Batch size
        validation_split=0.2,     # Use 20% of training for validation
        callbacks=[early_stop, reduce_lr],  # Use callbacks
        verbose=1                 # Show progress
    )
    
    print("="*70)
    print("TRAINING COMPLETED!")
    print("="*70)


# =====================================================================
# CELL 18: Plot Training History (Loss)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    plt.figure(figsize=(14, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss (MSE)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Mean Absolute Error', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/keras_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Training plots saved!")


# =====================================================================
# CELL 19: Evaluate Model on Test Set
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Evaluate using Keras evaluate method
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print("="*70)


# =====================================================================
# CELL 20: Make Predictions on Test Set
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Predict using Keras predict method
    y_pred = model.predict(X_test_scaled, verbose=0)
    
    print(f"Predictions made on {len(y_pred)} test samples")


# =====================================================================
# CELL 21: Calculate R² Score
# =====================================================================

from sklearn.metrics import r2_score

if TENSORFLOW_AVAILABLE:
    # Calculate R² score
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test R² Score: {r2:.4f}")


# =====================================================================
# CELL 22: Calculate Additional Metrics
# =====================================================================

from sklearn.metrics import mean_squared_error, mean_absolute_error

if TENSORFLOW_AVAILABLE:
    # Calculate metrics using sklearn
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\nAdditional Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")


# =====================================================================
# CELL 23: Plot Actual vs Predicted
# =====================================================================

if TENSORFLOW_AVAILABLE:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=30)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price', fontsize=12)
    plt.ylabel('Predicted Price', fontsize=12)
    plt.title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/claude/keras_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Actual vs Predicted plot saved!")


# =====================================================================
# CELL 24: Show Sample Predictions
# =====================================================================

if TENSORFLOW_AVAILABLE:
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (First 10 Test Samples)")
    print("="*70)
    print(f"{'#':<4} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print("-" * 70)
    
    for i in range(10):
        actual = y_test[i]
        predicted = y_pred[i][0]
        error = abs(actual - predicted)
        print(f"{i+1:<4} {actual:<12.4f} {predicted:<12.4f} {error:<12.4f}")


# =====================================================================
# CELL 25: Save Model (Keras format)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Save complete model
    model.save('/home/claude/keras_regression_model.h5')
    
    print("\nModel saved: keras_regression_model.h5")


# =====================================================================
# CELL 26: Save Model (SavedModel format - recommended)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Save in TensorFlow SavedModel format
    model.save('/home/claude/keras_model_savedformat')
    
    print("Model saved in SavedModel format: keras_model_savedformat/")


# =====================================================================
# CELL 27: Save Scaler
# =====================================================================

import pickle

# Save scaler
with open('/home/claude/keras_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler saved: keras_scaler.pkl")


# =====================================================================
# CELL 28: Load Saved Model (H5 format)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Load model from H5 file
    loaded_model = keras.models.load_model('/home/claude/keras_regression_model.h5')
    
    print("\nModel loaded from H5 file successfully!")


# =====================================================================
# CELL 29: Load Saved Model (SavedModel format)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Load model from SavedModel format
    loaded_model_2 = keras.models.load_model('/home/claude/keras_model_savedformat')
    
    print("Model loaded from SavedModel format successfully!")


# =====================================================================
# CELL 30: Load Scaler
# =====================================================================

# Load scaler
with open('/home/claude/keras_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

print("Scaler loaded successfully!")


# =====================================================================
# CELL 31: Prepare New/Unseen Data
# =====================================================================

# Select 5 random samples from test set (simulating new data)
np.random.seed(200)
new_indices = np.random.choice(len(X_test), size=5, replace=False)
new_data = X_test[new_indices]
new_actual = y_test[new_indices]

print(f"Selected {len(new_data)} new samples for prediction")


# =====================================================================
# CELL 32: Scale New Data
# =====================================================================

# Scale using loaded scaler
new_data_scaled = loaded_scaler.transform(new_data)

print("New data scaled!")


# =====================================================================
# CELL 33: Predict on New Data (using loaded model)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Predict using loaded model
    new_predictions = loaded_model.predict(new_data_scaled, verbose=0)
    
    print("Predictions made on new data!")


# =====================================================================
# CELL 34: Display New Data Predictions
# =====================================================================

if TENSORFLOW_AVAILABLE:
    print("\n" + "="*70)
    print("PREDICTIONS ON NEW/UNSEEN DATA")
    print("="*70)
    print(f"{'#':<4} {'Actual Price':<18} {'Predicted Price':<18} {'Error':<12}")
    print("-" * 70)
    
    for i in range(len(new_data)):
        actual = new_actual[i]
        predicted = new_predictions[i][0]
        error = abs(actual - predicted)
        
        print(f"{i+1:<4} ${actual*100:.2f}k{'':<10} ${predicted*100:.2f}k{'':<10} ${error*100:.2f}k")
    
    # Calculate R² on new samples
    new_r2 = r2_score(new_actual, new_predictions)
    print(f"\nR² Score on new samples: {new_r2:.4f}")


# =====================================================================
# CELL 35: Get Model Weights (for inspection)
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Get all layer weights
    model_weights = loaded_model.get_weights()
    
    print("\nModel has", len(model_weights), "weight matrices")
    for i, weight in enumerate(model_weights):
        print(f"  Layer {i//2 + 1} - {'Weights' if i % 2 == 0 else 'Biases'}: {weight.shape}")


# =====================================================================
# CELL 36: Get Model Configuration
# =====================================================================

if TENSORFLOW_AVAILABLE:
    # Get model config
    config = loaded_model.get_config()
    
    print("\nModel Configuration:")
    print(f"  Model type: {type(loaded_model).__name__}")
    print(f"  Number of layers: {len(loaded_model.layers)}")


# =====================================================================
# CELL 37: Visualize Model Architecture
# =====================================================================

if TENSORFLOW_AVAILABLE:
    try:
        # Try to plot model (requires pydot and graphviz)
        keras.utils.plot_model(
            loaded_model, 
            to_file='/home/claude/model_architecture.png',
            show_shapes=True,
            show_layer_names=True
        )
        print("Model architecture diagram saved!")
    except:
        print("Model architecture diagram could not be created (pydot/graphviz not available)")


# =====================================================================
# CELL 38: FINAL SUMMARY
# =====================================================================

if TENSORFLOW_AVAILABLE:
    print("\n" + "="*70)
    print("COMPLETE KERAS/TENSORFLOW REGRESSION IMPLEMENTATION")
    print("="*70)
    print(f"✓ Dataset: Housing Price Prediction")
    print(f"✓ Total Samples: {len(df)}")
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Training Samples: {len(X_train)}")
    print(f"✓ Test Samples: {len(X_test)}")
    print(f"✓ Model: Sequential Neural Network")
    print(f"✓ Architecture: [8 → 64 → 32 → 16 → 1]")
    print(f"✓ Optimizer: Adam")
    print(f"✓ Loss Function: MSE")
    print(f"✓ Training Epochs: {len(history.history['loss'])}")
    print(f"✓ Test MSE: {mse:.4f}")
    print(f"✓ Test RMSE: {rmse:.4f}")
    print(f"✓ Test MAE: {mae:.4f}")
    print(f"✓ Test R²: {r2:.4f}")
    print(f"✓ Model Saved: keras_regression_model.h5")
    print(f"✓ Model Saved: keras_model_savedformat/")
    print(f"✓ Scaler Saved: keras_scaler.pkl")
    print(f"✓ New Data Predictions: R² = {new_r2:.4f}")
    print(f"✓ Visualizations: 2 plots saved")
    print("="*70)
    print("\n✅ PURE KERAS IMPLEMENTATION COMPLETED!")
    print("✅ NO CUSTOM FUNCTIONS - ALL INBUILT METHODS")
    print("="*70)
else:
    print("\nTensorFlow not available. Please install it to run this code.")
    print("Run: pip install tensorflow")
