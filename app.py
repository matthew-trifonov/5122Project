import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
import calendar
import plotly.express as px
warnings.filterwarnings("ignore")

@st.cache_data
def preprocess(df):
    df['date'] = pd.to_datetime(df[['year','month', 'day']])
    df['week_number'] = df['date'].dt.strftime('%U')
    df['dow_name'] = df['date'].dt.strftime('%A')

    def season_cat(x):
        if x in [12, 1, 2]:
            return 'winter'
        elif x in [3, 4, 5]:
            return 'spring'
        elif x in [6, 7, 8]:
            return 'summer'
        return 'autumn'

    df['season'] = df['month'].apply(season_cat)
    df = df.drop(columns='year')

    # Check for origin and destination airports
    is_numeric = pd.to_numeric(df['origin_airport'], errors='coerce').notna() | pd.to_numeric(df['destination_airport'], errors='coerce').notna()
    df = df[~is_numeric]

    # Convert float times to proper time format
    def convert_float_time(float_time):
        time_string = str(float_time).split('.')[0].zfill(4)
        try:
            hour = int(time_string[:-2])
            minute = int(time_string[-2:])
            if hour >= 24:
                hour = hour % 24
            return f"{hour:02d}:{minute:02d}:00"
        except ValueError:
            return None

    # Apply conversion function
    df['arrival_time'] = df['arrival_time'].apply(convert_float_time)
    df['departure_time'] = df['departure_time'].apply(convert_float_time)
    df['scheduled_departure'] = df['scheduled_departure'].apply(convert_float_time)

    # Create a combined datetime column
    df['scheduled_departure_datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['scheduled_departure'])
    df['sched_hour'] = df['scheduled_departure_datetime'].dt.strftime('%H')
    
    return df

@st.cache_data
def load_data():
    airlines = pd.read_csv("airlines.csv")
    airlines = airlines.rename(columns={'AIRLINE': 'AL_FULLNAME', 'IATA_CODE': 'AIRLINE'})
    airports = pd.read_csv("airports.csv")
    flights = pd.read_csv("flights.csv")
    flights.columns = flights.columns.str.lower()
    airlines.columns = airlines.columns.str.lower()
    airports.columns = airports.columns.str.lower()
    flights_merged = flights.merge(airlines, how='left', on='airline')
    merged_df = pd.merge(flights_merged, airports, left_on='origin_airport', right_on='iata_code', how='left')
    merged_df.rename(columns={'airport': 'origin_airport_name', 'city': 'origin_city', 'state': 'origin_state'}, inplace=True)
    merged_df = pd.merge(merged_df, airports, left_on='destination_airport', right_on='iata_code', how='left')
    merged_df.rename(columns={'airport': 'destination_airport_name', 'city': 'destination_city', 'state': 'destination_state'}, inplace=True)
     
    df = preprocess(merged_df)
    
    # Decrease sample size to speed up rendering in streamlit when testing
    df = df.sample(frac=0.01, random_state=42)
    return df

# Use a smaller sample of data to speed things up
st.set_page_config(layout="wide")
st.title('US Flight Data')

df = load_data()

# Month and day name mappings
month_name_map = {i: calendar.month_name[i] for i in range(1, 13)}
day_name_map = {i: calendar.day_name[i] for i in range(7)}

global_filter = st.selectbox('Choose a global filter', ['Month', 'Weekday', 'Airline']).lower()

# Convert to Python datetime
min_date = df['date'].min().to_pydatetime() 
max_date = df['date'].max().to_pydatetime()

# Create date sidebar slider
start_date, end_date = st.sidebar.slider(
    'Select Date Range',
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Filter the data based on the selected date range
filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# Set filter values based on the global_filter selection
if global_filter == 'month':
    filter_values = filtered_df['month'].dropna().unique()
    filter_values = [month_name_map[i] for i in filter_values]
    filter_values.sort(key=lambda x: list(month_name_map.values()).index(x))
elif global_filter == 'weekday':
    filter_values = filtered_df['dow_name'].dropna().unique()
    filter_values = list(day_name_map.values())
    filter_values.sort(key=lambda x: list(day_name_map.values()).index(x))
elif global_filter == 'airline':
    filter_values = filtered_df['al_fullname'].dropna().unique()

# Sub filter
filter_value = st.selectbox(f'Select {global_filter} to filter by', filter_values)

# Apply filtering based on selected global filter
if global_filter == 'month':
    # Map the selected month name to its respective number
    month_number = [k for k, v in month_name_map.items() if v == filter_value][0]
    filtered_df = filtered_df[filtered_df['month'] == month_number]
    count_data = filtered_df.groupby('al_fullname').size().reset_index(name='count')
elif global_filter == 'weekday':
    filtered_df = filtered_df[filtered_df['dow_name'] == filter_value]
    count_data = filtered_df.groupby('al_fullname').size().reset_index(name='count')
elif global_filter == 'airline':
    filtered_df = filtered_df[filtered_df['al_fullname'] == filter_value]
    count_data = filtered_df.groupby('month').size().reset_index(name='count')

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6)) 
if global_filter == 'airline':
    # Bar plot
    count_data = filtered_df.groupby('month').size().reset_index(name='count')
    count_data['month'] = count_data['month'].map(month_name_map)
    
    sns.barplot(x='count', y='month', data=count_data, orient='h', palette="viridis", ax=axes[0])
    axes[0].set_xlabel('Number of Flights')
    axes[0].set_ylabel('Month')
    axes[0].set_title(f'Flights for {filter_value} across Months')

    # Pie chart
    pie_data = count_data.set_index('month')
    axes[1].pie(pie_data['count'], labels=pie_data.index, autopct='%1.1f%%', 
                colors=sns.color_palette("viridis", len(pie_data)), startangle=90)
    axes[1].set_title(f'Monthly Distribution for Airline {filter_value}')

else:
    # Bar plot
    count_data = filtered_df.groupby('al_fullname').size().reset_index(name='count')
    
    sns.barplot(x='count', y='al_fullname', data=count_data, orient='h', palette="viridis", ax=axes[0])
    axes[0].set_xlabel('Number of Flights')
    axes[0].set_ylabel('Airline')
    axes[0].set_title(f'Count of flights for {global_filter} = {filter_value}')

    pie_data = count_data.set_index('al_fullname')

    # Map full names to abbreviations for readability
    pie_labels = pie_data.index.map(lambda full_name: filtered_df[filtered_df['al_fullname'] == full_name]['airline'].iloc[0])
    
    axes[1].pie(pie_data['count'], labels=pie_labels, autopct='%1.1f%%', 
                colors=sns.color_palette("viridis", len(pie_data)), startangle=90)
    axes[1].set_title(f'Flight Distribution for {global_filter} = {filter_value}')


# Display both plots in Streamlit
st.pyplot(fig)

# Count flights by state
flight_counts_by_state = filtered_df.groupby('origin_state').size().reset_index(name='flight_count')

# Use Plotly to create a choropleth map
map = px.choropleth(flight_counts_by_state,
                    locations='origin_state',
                    locationmode='USA-states',
                    color='flight_count',
                    hover_name='origin_state',
                    color_continuous_scale='Viridis',
                    title="Flight Count by State (Origin)",
                    labels={'flight_count': 'Number of Flights'},
                    scope="usa")

# Update layout to make the map bigger
map.update_layout(
    width=1200,
    height=700,
    title_x=0.5,
    geo=dict(showcoastlines=True, coastlinecolor="Black", projection_type="albers usa")
)

# Display the map in Streamlit
st.plotly_chart(map)

# Show the filtered data (first few rows)
st.write(filtered_df.head())