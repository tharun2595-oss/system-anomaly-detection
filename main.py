import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Step 1: Create sample log data
np.random.seed(42)
data_size = 200

# Normal behavior
normal_data = np.random.normal(loc=50, scale=5, size=data_size)

# Add anomalies
anomalies = np.random.uniform(low=80, high=100, size=10)

# Combine
values = np.concatenate([normal_data, anomalies])

df = pd.DataFrame({
    'cpu_usage': values,
    'memory_usage': np.random.normal(60, 10, len(values)),
    'response_time': np.random.normal(200, 50, len(values))
})

# Step 2: Train model
model = IsolationForest(contamination=0.05)

# Train model
model.fit(df)

# Get predictions
predictions = model.predict(df)

# Create anomaly column FIRST
df['anomaly'] = predictions

# Step 3: Mark anomalies
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# Step 4: Plot
plt.plot(df['cpu_usage'], label='CPU Usage')

plt.scatter(
    df[df['anomaly']==1].index,
    df[df['anomaly']==1]['cpu_usage'],
    label='Anomaly'
)

plt.legend()
plt.show()