import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_absolute_error


# df = pd.read_csv('dataset.csv') 
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 3)  # 3个特征：cum_pop, cum_gdp, cum_edu
y = 10 * X[:,0] + 5 * X[:,1] - 3 * X[:,2] + np.random.normal(0, 2, n_samples)

# normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# model built
def build_meteorite_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)  # activation for regression
    
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae']
    )
    return model

model = build_meteorite_model(X_train.shape[1])
model.summary()

# 3.train model ------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 4. evaluate model ------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    
    print(f"Test R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"Prediction Examples:\n{np.stack([y_test[:5], y_pred[:5]], axis=1)}")

evaluate_model(model, X_test, y_test)

# 5. testing
new_data = np.array([[1.2, 0.8, -0.5]])  #u can change it
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled).flatten()[0]
print(f"\nPredicted meteorite count: {prediction:.2f}")
