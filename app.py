import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Visualization Dashboard", layout="wide")

def load_data(file):
    # Determine file type and read accordingly
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")
    return df

def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns

def get_categorical_columns(df):
    return df.select_dtypes(include=['object']).columns

def create_correlation_matrix(df, numeric_cols):
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns)
    return fig

def main():
    st.title("Interactive Data Visualization Dashboard")
    
    # Sidebar
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your data file (CSV or Excel)", 
                                           type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = load_data(uploaded_file)
            
            # Display basic information
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Rows", df.shape[0])
            with col2:
                st.metric("Number of Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())
            
            # Data preview
            with st.expander("Preview Data"):
                st.dataframe(df.head())
            
            # Get column types
            numeric_cols = get_numeric_columns(df)
            categorical_cols = get_categorical_columns(df)
            
            # Main visualization options
            st.subheader("Create Visualizations")
            
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Distribution Analysis", "Relationship Analysis", "Time Series Analysis"]
            )
            
            if viz_type == "Distribution Analysis":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Numeric Distribution")
                    num_col = st.selectbox("Select Numeric Column", numeric_cols)
                    
                    # Create distribution plot
                    fig = px.histogram(df, x=num_col, nbins=30)
                    st.plotly_chart(fig)
                    
                    # Show summary statistics
                    st.write("Summary Statistics:")
                    st.write(df[num_col].describe())
                
                with col2:
                    st.subheader("Categorical Distribution")
                    if len(categorical_cols) > 0:
                        cat_col = st.selectbox("Select Categorical Column", categorical_cols)
                        
                        # Create bar plot
                        value_counts = df[cat_col].value_counts()
                        fig = px.bar(x=value_counts.index, 
                                   y=value_counts.values,
                                   labels={'x': cat_col, 'y': 'Count'})
                        st.plotly_chart(fig)
            
            elif viz_type == "Relationship Analysis":
                st.subheader("Correlation Analysis")
                
                # Correlation matrix
                if len(numeric_cols) > 1:
                    fig = create_correlation_matrix(df, numeric_cols)
                    st.plotly_chart(fig)
                
                # Scatter plot
                st.subheader("Scatter Plot")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("Select X-axis", numeric_cols)
                with col2:
                    y_col = st.selectbox("Select Y-axis", numeric_cols)
                
                color_col = st.selectbox("Select Color Variable (optional)", 
                                       ['None'] + list(df.columns))
                
                if color_col == 'None':
                    fig = px.scatter(df, x=x_col, y=y_col)
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
                st.plotly_chart(fig)
            
            elif viz_type == "Time Series Analysis":
                st.subheader("Time Series Plot")
                
                # Check for datetime columns
                date_cols = df.select_dtypes(include=['datetime64']).columns
                if len(date_cols) == 0:
                    st.warning("No datetime columns found. Please make sure your date columns are in datetime format.")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        date_col = st.selectbox("Select Date Column", date_cols)
                    with col2:
                        value_col = st.selectbox("Select Value Column", numeric_cols)
                    
                    # Create time series plot
                    fig = px.line(df, x=date_col, y=value_col)
                    st.plotly_chart(fig)
            
            # Download processed data
            st.subheader("Download Processed Data")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your input data and try again.")

if __name__ == "__main__":
    main()
