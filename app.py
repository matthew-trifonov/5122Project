import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import calendar
import plotly.express as px
warnings.filterwarnings("ignore")
import openai
from dotenv import load_dotenv
import os
import streamlit as st
import base64


load_dotenv()
client = openai.OpenAI(
    base_url=os.getenv("GROQ_BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)


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

    airports = pd.read_csv("airports.csv")
    # Decrease sample size to speed up rendering in streamlit when testing
    df = df.sample(frac=0.01, random_state=42)
    return df
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
# Use a smaller sample of data to speed things up
st.set_page_config(layout="wide")
st.title('INSIGHTS: Airline Data Analysis')
tab1, tab2, tab3 = st.tabs(["Data plotting", "Data", "Insights Generation"])
df = load_data()
with tab1:
    img_base64 = get_base64_of_image("background.png")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
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
    st.markdown("---")
    st.subheader("📊 Bar and Pie Charts")
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
    st.markdown("---")
    st.subheader("🗺️ Choropleth Map: Flight Count by State")
    st.plotly_chart(map)

    abbr_companies = dict(zip(df['airline'], df['al_fullname']))

    # Define delay levels: 0 = on time, 1 = small delay, 2 = large delay
    delay_type = lambda x: ((0, 1)[x > 5], 2)[x > 45]
    df['DELAY_LEVEL'] = df['departure_delay'].apply(delay_type)

    # Create plot
    fig = plt.figure(figsize=(10, 7))
    ax = sns.countplot(y="airline", hue='DELAY_LEVEL', data=df)

    # Replace airline abbreviations with full names in y-axis
    labels = [abbr_companies.get(item.get_text(), item.get_text()) for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels)

    # Style plot
    plt.setp(ax.get_xticklabels(), fontsize=12, weight='normal')
    plt.setp(ax.get_yticklabels(), fontsize=12, weight='bold')
    ax.yaxis.label.set_visible(False)
    plt.xlabel('Flight Count', fontsize=16, weight='bold', labelpad=10)

    # Set legend labels
    L = plt.legend(title='Delay Level')
    L.get_texts()[0].set_text('on time (t < 5 min)')
    L.get_texts()[1].set_text('small delay (5 < t < 45 min)')
    L.get_texts()[2].set_text('large delay (t > 45 min)')

    plt.tight_layout()

    # ✅ Display in Streamlit
    st.subheader("✈️ Flight Delay Level Breakdown by Airline")
    st.pyplot(fig)
    
    

with tab2:
    # Month and day name mappings
    # Show the filtered data (first few rows)
    st.markdown("---")
    st.subheader("📄 Preview of Filtered Data")
    st.write(filtered_df.head())
    st.markdown("---")
    

with tab3:
    # Unified dropdown for insight generation
    st.markdown("---")
    st.subheader("📊 Generate Insight for Selected Visualization")
    visualization_option = st.selectbox(
        "Choose a visualization to analyze:",
        ["Bar Chart", "Pie Chart", "Choropleth Map"]
    )

    # Session storage for insight and chat
    if "insight_text" not in st.session_state:
        st.session_state.insight_text = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Insight generation section
    if st.button("Generate Insight"):
        if visualization_option == "Bar Chart":
            selected_table = count_data.to_markdown(index=False)
            prompt = f"""Below is a table summarizing flight counts grouped by {"month" if global_filter == "airline" else "airline"} with respect to {global_filter} = {filter_value}:

    {selected_table}

    Please provide a concise and insightful interpretation of the bar chart data. Identify patterns, noteworthy trends, and any surprising findings."""
        
        elif visualization_option == "Pie Chart":
            selected_table = count_data.to_markdown(index=False)
            prompt = f"""Below is a table showing the same data used in the pie chart visualization for {global_filter} = {filter_value}:

    {selected_table}

    Please provide a concise and insightful interpretation of the pie chart's distribution, highlighting notable proportions or dominance."""
        
        elif visualization_option == "Choropleth Map":
            selected_table = flight_counts_by_state.to_markdown(index=False)
            prompt = f"""I have created a choropleth map showing the number of flights originating from each U.S. state:\n\n{selected_table}\n
    Please analyze the distribution of flight counts across states, identify which states dominate, note regional patterns, and provide potential reasons for the distribution."""

        with st.spinner("Generating insight..."):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a data analyst interpreting airline data visualizations."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.session_state.insight_text = response.choices[0].message.content
            st.session_state.chat_history = []  # Reset chat
            st.session_state.insight_prompt = prompt 


    # Single-question chat interface (no history)
    if st.session_state.insight_text:
        st.markdown("#### ✨ Insight:")
        st.markdown(f"<div style='color: white'>{st.session_state.insight_text}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("💬 Ask a Follow-up Question")

        # Use form to avoid full rerun flicker
        with st.form(key="chat_form"):
            user_question = st.text_input("Ask a follow-up question:", key="chat_input")
            ask_btn = st.form_submit_button("💬 Ask")

        if ask_btn:
            if user_question and "insight_prompt" in st.session_state:
                with st.spinner("Getting answer..."):
                    messages = [
                        {"role": "system", "content": "You are a helpful data analyst interpreting airline visualizations."},
                        {"role": "user", "content": st.session_state.insight_prompt},
                        {"role": "user", "content": f"My follow-up question is: {user_question}"}
                    ]
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages
                    )
                    answer = response.choices[0].message.content

                # Show only the latest Q&A
                st.markdown(f"<div style='color: white'>**You asked:**</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color: white'>{user_question}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color: white'>**AI response:**</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color: white'>{answer}</div>", unsafe_allow_html=True)