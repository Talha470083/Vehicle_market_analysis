import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Vehicle Market Analysis",
    page_icon="🚗",
    layout="wide"
)

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\User\Desktop\IDS\cleaned_vehicles.csv")

df = load_data()

# ===== UTILITY FUNCTIONS =====
def create_stats_table(data, column_name, is_currency=False, is_year=False):
    """Generate a statistics table for a given column"""
    stats = {
        'Statistic': ['Mean', 'Median', 'Mode', 'Minimum', 'Maximum', 
                     'Std Deviation', '25th Percentile', '75th Percentile', 'IQR',
                     'Total Count', 'Non-Zero Count'],
        'Value': [
            data[column_name].mean(),
            data[column_name].median(),
            data[column_name].mode()[0],
            data[column_name].min(),
            data[column_name].max(),
            data[column_name].std(),
            data[column_name].quantile(0.25),
            data[column_name].quantile(0.75),
            data[column_name].quantile(0.75) - data[column_name].quantile(0.25),
            len(data),
            (data[column_name] > 0).sum()
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    
    if is_currency:
        stats_df.loc[stats_df['Statistic'].isin(['Mean', 'Median', 'Mode', 'Minimum', 'Maximum', 
                                               '25th Percentile', '75th Percentile', 'IQR', 'Std Deviation']), 
                    'Value'] = stats_df.loc[stats_df['Statistic'].isin(['Mean', 'Median', 'Mode', 'Minimum', 'Maximum', 
                                                                      '25th Percentile', '75th Percentile', 'IQR', 'Std Deviation']), 
                                          'Value'].apply(lambda x: f"${x:,.2f}")
    elif is_year:
        stats_df.loc[stats_df['Statistic'].isin(['Mean', 'Median', 'Mode', 'Minimum', 'Maximum']), 
                    'Value'] = stats_df.loc[stats_df['Statistic'].isin(['Mean', 'Median', 'Mode', 'Minimum', 'Maximum']), 
                                          'Value'].apply(lambda x: f"{int(x)}")
    else:
        stats_df.loc[stats_df['Statistic'].isin(['Mean', 'Median', 'Mode', 'Minimum', 'Maximum', 
                                               '25th Percentile', '75th Percentile', 'IQR', 'Std Deviation']), 
                    'Value'] = stats_df.loc[stats_df['Statistic'].isin(['Mean', 'Median', 'Mode', 'Minimum', 'Maximum', 
                                                                      '25th Percentile', '75th Percentile', 'IQR', 'Std Deviation']), 
                                          'Value'].apply(lambda x: f"{x:,.0f}")

    return stats_df

# ===== SIDEBAR FILTERS =====
with st.sidebar:
    st.header("🔍 Data Filters")
    
    # Manufacturer filter
    manufacturers = sorted(df['manufacturer'].unique())
    selected_manufacturers = st.multiselect(
        "Select manufacturers",
        options=manufacturers,
        default=manufacturers[:3]
    )
    
    # Condition filter
    conditions = sorted(df['condition'].unique())
    selected_conditions = st.multiselect(
        "Select conditions",
        options=conditions,
        default=conditions[:2]
    )
    
    # Price range slider
    price_min, price_max = int(df['price'].min()), int(df['price'].max())
    price_range = st.slider(
        "Price range (USD)",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max)
    )
    
    # Year range slider
    if 'year' in df.columns:
        year_min, year_max = int(df['year'].min()), int(df['year'].max())
        year_range = st.slider(
            "Manufacturing year range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max)
        )

# Apply filters
filtered_df = df.copy()
if selected_manufacturers:
    filtered_df = filtered_df[filtered_df['manufacturer'].isin(selected_manufacturers)]
if selected_conditions:
    filtered_df = filtered_df[filtered_df['condition'].isin(selected_conditions)]
filtered_df = filtered_df[filtered_df['price'].between(price_range[0], price_range[1])]
if 'year' in df.columns:
    filtered_df = filtered_df[filtered_df['year'].between(year_range[0], year_range[1])]

# ===== MAIN APP =====
st.title("🚗 Vehicle Market Analysis Dashboard")

# Navigation buttons
tab1, tab2 , tab3 , tab4 = st.tabs(["ℹ️ Introduction" , "📊 EDA Section", "🤖 Model Section", "📌 Conclusion"])

with tab1:  # Introduction tab
    st.header("Project Introduction")
    st.markdown("""
    ### Vehicle Market Analysis Project
    
    **Objective**: Analyze vehicle listing data to identify market trends, pricing factors, and key insights.
    
    **Dataset Overview**:
    - Contains {:,} vehicle listings
    - Features include price, year, odometer, condition, manufacturer, and more
    - Data cleaned and preprocessed for analysis
    
    **Project Goals**:
    1. Perform comprehensive exploratory data analysis (EDA)
    2. Identify key pricing factors
    3. Build predictive models
    4. Create interactive visualizations
    """.format(len(df)))
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:  # EDA Section
    st.header("Exploratory Data Analysis")
    
    # Row 1: Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Listings", f"{len(filtered_df):,}")
    with col2:
        st.metric("Average Price", f"${filtered_df['price'].mean():,.2f}")
    with col3:
        st.metric("Average Odometer", f"{filtered_df['odometer'].mean():,.0f} miles")
    
    # Row 2: Price and Odometer Stats
    st.subheader("Price Statistics")
    price_stats = create_stats_table(filtered_df, 'price', is_currency=True)
    st.dataframe(price_stats, use_container_width=True, hide_index=True)
    
    st.subheader("Odometer Statistics")
    odo_stats = create_stats_table(filtered_df, 'odometer', is_currency=False)
    st.dataframe(odo_stats, use_container_width=True, hide_index=True)
 
    # Row 3: Distribution Plots
    st.subheader("Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Price Distribution
    sns.histplot(filtered_df['price'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("Price Distribution")
    axes[0].set_xlabel("Price ($)")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    # Odometer Distribution
    sns.histplot(filtered_df['odometer'], kde=True, ax=axes[1], color='salmon')
    axes[1].set_title("Odometer Distribution")
    axes[1].set_xlabel("Mileage")
    
    # Year Distribution
    if 'year' in filtered_df.columns:
        sns.histplot(filtered_df['year'], kde=True, ax=axes[2], color='lightgreen')
        axes[2].set_title("Year Distribution")
        axes[2].set_xlabel("Year")
    
    st.pyplot(fig)
    
    # Row 4: Top Manufacturers
    st.subheader("Manufacturer Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Count by manufacturer
        manufacturer_counts = filtered_df['manufacturer'].value_counts().head()
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=manufacturer_counts.values, y=manufacturer_counts.index, palette="viridis")
        plt.title("Top 15 Manufacturers by Count")
        plt.xlabel("Number of Listings")
        st.pyplot(fig)
    
    with col2:
        # Average price by manufacturer
        if not filtered_df.empty:
            avg_prices = filtered_df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(15)
            fig = plt.figure(figsize=(10, 6))
            sns.barplot(x=avg_prices.values, y=avg_prices.index, palette="plasma")
            plt.title("Top 15 Manufacturers by Average Price")
            plt.xlabel("Average Price ($)")
            plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
            st.pyplot(fig)
    
    # Row 5: Condition and Fuel Type
    st.subheader("Categorical Distributions")
    col1, col2 = st.columns(2)
    
    with col1:
        # Condition distribution
        condition_counts = filtered_df['condition'].value_counts()
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=condition_counts.index, y=condition_counts.values, palette="Blues_r")
        plt.title("Vehicle Condition Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Fuel type distribution
        if 'fuel' in filtered_df.columns:
            fuel_counts = filtered_df['fuel'].value_counts()
            fig = plt.figure(figsize=(10, 6))
            sns.barplot(x=fuel_counts.index, y=fuel_counts.values, palette="Greens_r")
            plt.title("Fuel Type Distribution")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    # Row 6: Correlation Matrix
    st.subheader("Feature Correlations")
    numerical_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cols_to_exclude = ['posting_year', 'posting_month', 'posting_day_of_week']
    numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude]
    
    if numerical_cols:
        corr_matrix = filtered_df[numerical_cols].corr()
        fig = plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title("Numerical Features Correlation Matrix")
        st.pyplot(fig)
    else:
        st.warning("No numerical columns available for correlation analysis")

with tab3:  # Model Section
    st.header("🚗 Vehicle Price Prediction Model")
    
    # Load required imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
    import os
    from sklearn.pipeline import Pipeline

    # Enhanced model loading function
    @st.cache_resource
    def load_model():
        try:
            model_path = os.path.join('models', 'vehicle_price_predictor.pkl')
            if not os.path.exists(model_path):
                st.error("Model file not found. Please train the model first.")
                return None, None, None, None
                
            model_data = joblib.load(model_path)
            model = model_data['model']
            
            # Get feature names from one-hot encoder
            numeric_features = ['year', 'odometer', 'age']
            categorical_features = ['condition', 'manufacturer', 'fuel', 'transmission']
            
            if isinstance(model, Pipeline) and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                if 'cat' in preprocessor.named_transformers_:
                    ohe = preprocessor.named_transformers_['cat']
                    cat_feature_names = []
                    for i, feature in enumerate(categorical_features):
                        categories = ohe.categories_[i]
                        cat_feature_names.extend([f"{feature}_{cat}" for cat in categories])
                    all_feature_names = numeric_features + cat_feature_names
                else:
                    all_feature_names = numeric_features + categorical_features
            else:
                all_feature_names = numeric_features + categorical_features
            
            return (
                model,
                model_data['metrics'],
                all_feature_names,
                model_data.get('timestamp', None)
            )
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None, None, None, None
    
    model, metrics, feature_names, timestamp = load_model()
    model_loaded = model is not None
    
    if model_loaded:
        # Display model information
        st.markdown("""
        ### Model Performance
        *Trained on vehicle market data with Random Forest Regressor*
        """)
        
        if timestamp:
            st.caption(f"Model trained on: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", 
                     f"{metrics['test']['r2']:.3f}",
                     help="Explained variance (0-1, higher is better)")
        with col2:
            st.metric("Mean Absolute Error", 
                     f"${metrics['test']['mae']:,.0f}",
                     help="Average prediction error in dollars")
        with col3:
            st.metric("Error Range", 
                     f"± ${metrics['test']['rmse']:,.0f}",
                     help="Typical prediction range")
        
        # Feature importance visualization
        st.subheader("📊 What Factors Affect Price Most?")
        try:
            if (model_loaded and 
                hasattr(model, 'named_steps') and 
                'regressor' in model.named_steps and 
                hasattr(model.named_steps['regressor'], 'feature_importances_')):
                
                importances = model.named_steps['regressor'].feature_importances_
                
                # Ensure we have matching feature names
                if len(feature_names) != len(importances):
                    st.warning(f"Feature count mismatch (names: {len(feature_names)}, importances: {len(importances)})")
                    feature_names = [f"Feature {i+1}" for i in range(len(importances))]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot top features
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=importance_df.head(10),
                    x='Importance',
                    y='Feature',
                    palette="viridis"
                )
                plt.title("Top 10 Price Influencers")
                plt.xlabel("Relative Importance")
                st.pyplot(fig)
                
                # Show complete feature importance
                with st.expander("View complete feature importance"):
                    st.dataframe(
                        importance_df,
                        column_config={
                            "Importance": st.column_config.ProgressColumn(
                                format="%.3f",
                                min_value=0,
                                max_value=importance_df['Importance'].max()
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
        except Exception as e:
            st.warning(f"Feature importance visualization unavailable: {str(e)}")
    
    # Prediction interface
    st.subheader("💵 Get a Price Estimate")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            manufacturer = st.selectbox(
                "Make",
                df['manufacturer'].unique(),
                help="Select vehicle manufacturer"
            )
            year = st.slider(
                "Year", 
                1950, 2023, 2015,
                help="Manufacturing year"
            )
            condition = st.selectbox(
                "Condition", 
                df['condition'].unique(),
                help="Vehicle condition rating"
            )
        
        with col2:
            odometer = st.number_input(
                "Mileage", 
                0, 500000, 50000, step=1000,
                help="Odometer reading in miles"
            )
            fuel_type = st.selectbox(
                "Fuel Type", 
                df['fuel'].unique() if 'fuel' in df.columns else ['gas'],
                help="Fuel type"
            )
            transmission = st.selectbox(
                "Transmission", 
                df['transmission'].unique() if 'transmission' in df.columns else ['automatic'],
                help="Transmission type"
            )
        
        submitted = st.form_submit_button("Calculate Price", type="primary")
        
        if submitted:
            if model_loaded:
                try:
                    # Create input data
                    input_data = pd.DataFrame({
                        'manufacturer': [manufacturer],
                        'year': [year],
                        'condition': [condition],
                        'odometer': [odometer],
                        'fuel': [fuel_type],
                        'transmission': [transmission],
                        'age': [2023 - year]
                    })
                    
                    # Make prediction
                    predicted_price = model.predict(input_data)[0]
                    
                    # Display results
                    st.success("### Price Estimation Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Predicted Value",
                            f"${predicted_price:,.0f}",
                            help="Model's best estimate"
                        )
                    
                    with col2:
                        st.metric(
                            "Expected Range",
                            f"${predicted_price - metrics['test']['rmse']:,.0f} - ${predicted_price + metrics['test']['rmse']:,.0f}",
                            help="Typical prediction range based on model accuracy"
                        )
                    
                    # Show similar vehicles
                    similar_vehicles = df[
                        (df['manufacturer'] == manufacturer) &
                        (df['year'].between(year-2, year+2)) &
                        (df['condition'] == condition)
                    ].sort_values('price')
                    
                    if not similar_vehicles.empty:
                        with st.expander("🔍 See Similar Vehicles in Dataset"):
                            st.dataframe(
                                similar_vehicles[['year', 'odometer', 'price']],
                                column_config={
                                    "year": "Year",
                                    "odometer": "Mileage",
                                    "price": st.column_config.NumberColumn(
                                        "Price",
                                        format="$%.0f"
                                    )
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Try adjusting your inputs or check for unusual values")
            
            else:
                # Fallback calculation
                base_price = 15000
                year_adj = (year - 2000) * 500
                odometer_adj = - (odometer / 10000) * 800
                predicted_price = base_price + year_adj + odometer_adj
                
                st.warning("⚠️ Using simplified estimation (model not loaded)")
                st.success(f"Estimated Price: ${predicted_price:,.0f}")
                
                st.info("""
                **For more accurate predictions:**
                - Ensure the model file exists in `models/vehicle_price_predictor.pkl`
                - Verify all dependencies are installed
                - Check the training script has been run
                """)
with tab4:  # Conclusion
    st.header("Project Conclusions")
    
    st.markdown("""
    ### Key Findings
    
    1. **Price Drivers**:
       - Vehicle year and odometer reading are the strongest predictors of price
       - Certain manufacturers command premium prices
       - Condition significantly affects resale value
    
    2. **Market Trends**:
       - SUVs and trucks dominate recent listings
       - Electric vehicle listings are growing rapidly
       - Average prices peak for 3-5 year old vehicles
    
    3. **Data Quality**:
       - Odometer readings showed some unrealistic values that were cleaned
       - Manufacturer names required standardization
    
    ### Recommendations
    
    - Focus inventory on late-model SUVs and trucks
    - Consider certified pre-owned programs for higher margins
    - Monitor electric vehicle market trends closely
    
    ### Limitations
    
    - Data limited to certain geographic regions
    - Some features had missing values
    - Model doesn't capture recent market disruptions
    """)
    
    st.subheader("Next Steps")
    st.markdown("""
    - Incorporate additional data sources (auction results, economic indicators)
    - Develop time-series forecasting models
    - Build dealer-specific pricing recommendations
    """)

# Add download button for filtered data
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_vehicles.csv",
    mime="text/csv"
)
