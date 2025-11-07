import pandas as pd
import numpy as np

# Create datetime range
dates = pd.date_range(start="2025-10-21", periods=168, freq='H')

# Generate synthetic patterns
data = {
    "DateTime": dates,
    "Global_active_power": 1.0 + 1.5*np.sin(2*np.pi*dates.hour/24) + np.random.normal(0, 0.1, len(dates)),
    "Global_reactive_power": 0.1 + 0.05*np.sin(2*np.pi*dates.hour/24) + np.random.normal(0, 0.01, len(dates)),
    "Voltage": 242 + np.random.normal(0, 0.2, len(dates)),
    "Sub_metering_1": np.random.randint(0, 2, len(dates)),
    "Sub_metering_2": np.random.randint(0, 2, len(dates)),
    "Sub_metering_3": np.random.randint(16, 22, len(dates))
}

df = pd.DataFrame(data)
df.to_csv("sample_energy_data.csv", index=False)

print("✅ Created sample_energy_data.csv with 7 days of realistic power readings")
