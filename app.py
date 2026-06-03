# Here we build the streamlit app.
# We will create a simple interface where users can input the performance metrics of 
# their YouTube video and get an estimated ad revenue prediction based on our trained model.


# Importing necessary libraries for the Streamlit app.

import streamlit as st    
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the trained model, scaler, and features list.
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

# Load the cleaned dataset for insights and visualizations.

@st.cache_data
def load_data():
    return pd.read_csv('cleaned_youtube_data.csv')

model, scaler, features = load_model()
df = load_data()

# Step 1:
# Now we creare the Streamlit app layout with a sidebar for navigation and different pages for home, prediction, and data insights.

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Predict Revenue", "Data Insights"]
)

#Step 1.1: HOME

if page == "Home":
    st.title("YouTube Content Monetization Modeler")
    st.subheader("Predict Ad Revenue for YouTube Videos")

    st.markdown("""
    ### What does this app do?
    This tool uses **Machine Learning** to predict how much
    ad revenue a YouTube video will earn based on its
    performance metrics.

    ### How to use:
    1. Go to **Predict Revenue** → enter your video stats
    2. Go to **Data Insights** → explore the data visually
    
    ### Model Performance:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="R² Score",
                  value="0.9526",
                  delta="95% accuracy")
    with col2:
        st.metric(label="RMSE",
                  value="$13.48",
                  delta="avg error in USD")
    with col3:
        st.metric(label="MAE",
                  value="$3.12",
                  delta="most predictions within $3")

    st.info("Use the sidebar to navigate!")

# Step 1.2: PREDICT REVENUE

elif page == "Predict Revenue":
    st.title("Predict Ad Revenue")
    st.write("Enter your video's performance metrics below:")

    col1, col2 = st.columns(2)

    with col1:
        views = st.number_input(
            "Views",
            min_value=0, max_value=100000,
            value=10000, step=100
        )
        likes = st.number_input(
            "Likes",
            min_value=0, max_value=100000,
            value=500, step=10
        )
        comments = st.number_input(
            "Comments",
            min_value=0, max_value=10000,
            value=50, step=5
        )
        watch_time = st.number_input(
            "Watch Time (minutes)",
            min_value=0, max_value=100000,
            value=30000, step=100
        )

    with col2:
        subscribers = st.number_input(
            "Subscribers",
            min_value=0, max_value=1000000,
            value=10000, step=1000
        )
        video_length = st.number_input(
            "Video Length (minutes)",
            min_value=1, max_value=60,
            value=10, step=1
        )
        category = st.selectbox(
            "Category",
            ["Education", "Entertainment",
             "Gaming", "Lifestyle", "Music", "Tech"]
        )
        device = st.selectbox(
            "Device",
            ["Desktop", "Mobile", "Tablet", "TV"]
        )
        country = st.selectbox(
            "Country",
            ["AU", "CA", "DE", "IN", "UK", "US"]
        )

    # Encoding maps
    category_map = {
        "Education": 0, "Entertainment": 1,
        "Gaming": 2, "Lifestyle": 3,
        "Music": 4, "Tech": 5
    }
    device_map = {
        "Desktop": 0, "Mobile": 1,
        "Tablet": 2, "TV": 3
    }
    country_map = {
        "AU": 0, "CA": 1, "DE": 2,
        "IN": 3, "UK": 4, "US": 5
    }

    if st.button("Predict Revenue", type="primary"):
        input_data = np.array([[
            watch_time,
            views,
            likes,
            comments,
            subscribers,
            video_length,
            category_map[category],
            device_map[device],
            country_map[country]
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction = max(128, min(380, prediction))

        st.success(f"Estimated Ad Revenue: **${prediction:.2f}**")

        st.write("### Your Video Stats")
        stats_df = pd.DataFrame({
            "Metric": ["Views", "Likes", "Comments",
                       "Watch Time (min)", "Subscribers"],
            "Value": [views, likes, comments,
                      watch_time, subscribers]
        })
        st.dataframe(stats_df)


# Step 1.3: DATA INSIGHTS

elif page == "Data Insights":
    st.title("Data Insights")
    st.write("Explore patterns in the YouTube dataset")

    # Dataset summary
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Videos", f"{len(df):,}")
    col2.metric("Avg Revenue", f"${df['ad_revenue_usd'].mean():.2f}")
    col3.metric("Max Revenue", f"${df['ad_revenue_usd'].max():.2f}")

    # Chart 1: Revenue Distribution
    st.subheader("Revenue Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.hist(df['ad_revenue_usd'], bins=50,
             color='steelblue', edgecolor='white')
    ax1.set_xlabel('Ad Revenue (USD)')
    ax1.set_ylabel('Count')
    ax1.set_title('How is revenue distributed?')
    st.pyplot(fig1)
    st.info("Revenue is uniformly spread between $130-$380")

    # Chart 2: Revenue by Category
    st.subheader("Revenue by Category")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    cat_rev = df.groupby('category')['ad_revenue_usd'].mean()
    cat_rev.sort_values(ascending=False).plot(
        kind='bar', ax=ax2, color='mediumpurple'
    )
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Average Revenue (USD)')
    ax2.set_title('Which content category earns most?')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Chart 3: Revenue by Device
    st.subheader("Revenue by Device")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    dev_rev = df.groupby('device')['ad_revenue_usd'].mean()
    dev_rev.sort_values(ascending=False).plot(
        kind='bar', ax=ax3, color='coral'
    )
    ax3.set_xlabel('Device')
    ax3.set_ylabel('Average Revenue (USD)')
    ax3.set_title('Which device generates more revenue?')
    plt.xticks(rotation=0)
    st.pyplot(fig3)