import pandas as pd
import numpy as np

print("\n🌱 Starting drought prediction pipeline...\n")

# =====================================================
# LOAD DATA
# =====================================================

climate_df = pd.read_csv('data/climate_data.csv')
climate_df['time'] = pd.to_datetime(climate_df['time'])

soil_df = pd.read_csv('data/soil_data.csv')
soil_df['Fecha'] = pd.to_datetime(soil_df['Fecha'])

# Ensure correct ordering
climate_df = climate_df.sort_values(['field', 'time']).reset_index(drop=True)

# =====================================================
# SPI CALCULATION
# =====================================================

def calculate_spi(precipitation_series, scale=30):
    spi_values = []
    for i in range(len(precipitation_series)):
        if i < scale:
            spi_values.append(np.nan)
        else:
            window = precipitation_series[i-scale:i]
            mean = np.mean(window)
            std = np.std(window)
            spi = (precipitation_series[i] - mean) / std if std > 0 else 0
            spi_values.append(spi)
    return spi_values

# =====================================================
# CLIMATE PROCESSING (ROLLING WEATHER CORRECTED)
# =====================================================

print("🌦 Calculating rolling weather + drought indices...")

climate_outputs = []

for field, field_data in climate_df.groupby('field'):

    field_data = field_data.sort_values('time').copy()

    # ✅ DAILY ROLLING WEATHER FEATURES (FIXED)
    field_data['precip_30day_sum'] = (
        field_data['precipitation_sum']
        .rolling(window=30, min_periods=30)
        .sum()
    )

    field_data['precip_90day_sum'] = (
        field_data['precipitation_sum']
        .rolling(window=90, min_periods=90)
        .sum()
    )

    avg_temp_daily = (
        field_data['temperature_2m_max'] +
        field_data['temperature_2m_min']
    ) / 2

    field_data['temp_30day_avg'] = (
        avg_temp_daily
        .rolling(window=30, min_periods=30)
        .mean()
    )

    # SPI
    field_data['SPI_30'] = calculate_spi(
        field_data['precipitation_sum'].values, scale=30
    )

    # Binary drought from SPI
    field_data['drought_binary_SPI'] = (field_data['SPI_30'] < -1).astype(int)

    # SPI severity numeric
    def drought_severity(spi):
        if pd.isna(spi):
            return 0
        elif spi >= -1.0:
            return 1   # Mild
        elif spi >= -1.5:
            return 2   # Moderate
        else:
            return 3   # Severe

    field_data['drought_severity'] = field_data['SPI_30'].apply(drought_severity)

    climate_outputs.append(field_data)

climate_final = pd.concat(climate_outputs, ignore_index=True)

# =====================================================
# SOIL DROUGHT CLASSIFICATION
# =====================================================

print("🌿 Processing soil drought indicators...")

def ndvi_based_drought(ndvi, gndvi, ndwi):
    if ndvi < 0.3:
        return 2   # Severe
    elif ndvi < 0.5:
        return 1   # Moderate
    else:
        return 0   # No drought

soil_df['drought_soil_based'] = soil_df.apply(
    lambda row: ndvi_based_drought(row['NDVI'], row['GNDVI'], row['NDWI']),
    axis=1
)

soil_df['drought_binary_soil'] = (soil_df['drought_soil_based'] > 0).astype(int)

# =====================================================
# MERGE CLIMATE + SOIL
# =====================================================

print("🔗 Merging climate and soil datasets...")

merged_rows = []

for field in climate_final['field'].unique():

    climate_field = climate_final[climate_final['field'] == field].copy()
    soil_field = soil_df[soil_df['Field'] == field].copy()

    for idx, row in climate_field.iterrows():
        date = row['time']
        soil_match = soil_field[
            abs((soil_field['Fecha'] - date).dt.days) <= 5
        ]

        if not soil_match.empty:
            closest_idx = abs((soil_match['Fecha'] - date).dt.days).idxmin()
            soil_row = soil_field.loc[closest_idx]

            climate_field.loc[idx, 'drought_soil_based'] = soil_row['drought_soil_based']
            climate_field.loc[idx, 'drought_binary_soil'] = soil_row['drought_binary_soil']

    merged_rows.append(climate_field)

all_predictions_merged = pd.concat(merged_rows, ignore_index=True)
all_predictions_merged.fillna(method='ffill', inplace=True)

# =====================================================
# FINAL DROUGHT LOGIC
# =====================================================

print("🧠 Computing final drought risk, severity and confidence...")

# Scale soil severity into climate scale
soil_severity_mapping = {0: 0, 1: 2, 2: 3}

all_predictions_merged['soil_severity_scaled'] = (
    all_predictions_merged['drought_soil_based']
    .map(soil_severity_mapping)
    .fillna(0)
)

# Worst-case severity numeric
all_predictions_merged['severity_numeric'] = all_predictions_merged[
    ['drought_severity', 'soil_severity_scaled']
].max(axis=1)

# Final severity label
def severity_label(value):
    if value <= 1:
        return 'Mild'
    elif value == 2:
        return 'Moderate'
    else:
        return 'Severe'

all_predictions_merged['severity'] = all_predictions_merged['severity_numeric'].apply(severity_label)

# Risk
all_predictions_merged['drought_risk'] = np.where(
    (all_predictions_merged['drought_binary_SPI'] == 1) |
    (all_predictions_merged['drought_binary_soil'] == 1),
    'High',
    'Low'
)

# Confidence score
def drought_confidence(row):
    score = 0
    if row['drought_binary_SPI'] == 1:
        score += 40
    if row['drought_binary_soil'] == 1:
        score += 40
    if row['severity'] == 'Moderate':
        score += 10
    elif row['severity'] == 'Severe':
        score += 20
    return min(score, 100)  

all_predictions_merged['drought_confidence'] = all_predictions_merged.apply(
    drought_confidence, axis=1
)

# =====================================================
# EXPORT FINAL CSV
# =====================================================

final_output = all_predictions_merged[
    [
        'time',
        'field',
        'precip_30day_sum',
        'precip_90day_sum',
        'temp_30day_avg',
        'drought_risk',
        'severity',
        'drought_confidence'
    ]
]

output_file = 'FINAL_DROUGHT_REPORT.csv'
final_output.to_csv(output_file, index=False)

print("\n✅ DONE!")
print(f"📄 Output saved as: {output_file}")
print("\nSample output:")
print(final_output.head(10))

