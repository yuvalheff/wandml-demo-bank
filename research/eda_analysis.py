import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Define color palette for consistent plotting
app_color_palette = [
    'rgba(99, 110, 250, 0.8)',   # Blue
    'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
    'rgba(0, 204, 150, 0.8)',    # Green
    'rgba(171, 99, 250, 0.8)',   # Purple
    'rgba(255, 161, 90, 0.8)',   # Orange
    'rgba(25, 211, 243, 0.8)',   # Cyan
    'rgba(255, 102, 146, 0.8)',  # Pink
    'rgba(182, 232, 128, 0.8)',  # Light Green
    'rgba(255, 151, 255, 0.8)',  # Magenta
    'rgba(254, 203, 82, 0.8)'    # Yellow
]

# Load the dataset
print("Loading dataset...")
train_data = pd.read_csv('/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/data/train.csv')

print(f"Dataset shape: {train_data.shape}")
print(f"Columns: {list(train_data.columns)}")
print(f"First few rows:")
print(train_data.head())

# Identify target column
target_col = 'target'
print(f"Target column: {target_col}")

# Check for missing values and data types
print("\n=== Data Quality Analysis ===")
missing_values = train_data.isnull().sum()
data_types = train_data.dtypes

data_quality_df = pd.DataFrame({
    'Data Type': data_types,
    'Missing Values': missing_values,
    'Missing %': (missing_values / len(train_data)) * 100
})

print("Data Quality Summary:")
print(data_quality_df)

# 1. Data Quality Overview Plot
fig = go.Figure(data=[go.Bar(
    x=data_quality_df.index,
    y=data_quality_df['Missing %'],
    marker_color=app_color_palette[0]
)])

fig.update_layout(
    height=550,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=12),
    title_font=dict(color='#7C3AED', size=16),
    xaxis=dict(
        title='Features',
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)',
        tickfont=dict(color='#8B5CF6', size=11),
        title_font=dict(color='#7C3AED', size=12)
    ),
    yaxis=dict(
        title='Missing Values (%)',
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)',
        tickfont=dict(color='#8B5CF6', size=11),
        title_font=dict(color='#7C3AED', size=12)
    ),
    legend=dict(font=dict(color='#8B5CF6', size=11))
)

fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/research/plots/data_quality_overview.html", 
               include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

print("\n=== Target Variable Analysis ===")
target_counts = train_data[target_col].value_counts()
target_pct = train_data[target_col].value_counts(normalize=True) * 100

print(f"Target variable '{target_col}' distribution:")
print(f"Counts:\n{target_counts}")
print(f"\nPercentages:\n{target_pct}")

# 2. Target Distribution Plot
fig = px.pie(values=target_counts.values, names=target_counts.index,
             color_discrete_sequence=app_color_palette[:2])

fig.update_layout(
    height=550,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=12),
    title_font=dict(color='#7C3AED', size=16),
    legend=dict(font=dict(color='#8B5CF6', size=11))
)

fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/research/plots/target_distribution.html", 
               include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# Separate numerical and categorical features
numerical_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

print(f"\nNumerical features ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# 3. Numerical Features Distribution
print("\n=== Numerical Features Analysis ===")
if numerical_cols:
    print("Basic statistics for numerical features:")
    print(train_data[numerical_cols].describe())
    
    # Create numerical distribution plots
    n_cols = len(numerical_cols)
    rows = (n_cols + 2) // 3
    cols = min(3, n_cols)
    
    fig = make_subplots(rows=rows, cols=cols, 
                        subplot_titles=numerical_cols)
    
    for i, col in enumerate(numerical_cols):
        row = (i // 3) + 1
        col_pos = (i % 3) + 1
        
        fig.add_trace(
            go.Histogram(x=train_data[col], name=col, 
                        marker_color=app_color_palette[i % len(app_color_palette)],
                        showlegend=False),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B5CF6', size=12),
        title_font=dict(color='#7C3AED', size=16)
    )
    
    fig.update_xaxes(gridcolor='rgba(139,92,246,0.2)', zerolinecolor='rgba(139,92,246,0.3)',
                     tickfont=dict(color='#8B5CF6', size=10),
                     title_font=dict(color='#7C3AED', size=11))
    fig.update_yaxes(gridcolor='rgba(139,92,246,0.2)', zerolinecolor='rgba(139,92,246,0.3)',
                     tickfont=dict(color='#8B5CF6', size=10),
                     title_font=dict(color='#7C3AED', size=11))
    
    fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/research/plots/numerical_distributions.html", 
                   include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# 4. Categorical Features Distribution
print("\n=== Categorical Features Analysis ===")
if categorical_cols:
    for col in categorical_cols:
        print(f"\n{col}: {train_data[col].nunique()} unique values")
        print(train_data[col].value_counts().head(10))
    
    # Plot top 4 categorical features
    key_categorical = categorical_cols[:4] if len(categorical_cols) >= 4 else categorical_cols
    
    if key_categorical:
        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=key_categorical,
                            specs=[[{"type": "xy"}, {"type": "xy"}],
                                   [{"type": "xy"}, {"type": "xy"}]])
        
        for i, col in enumerate(key_categorical):
            row = (i // 2) + 1
            col_pos = (i % 2) + 1
            
            value_counts = train_data[col].value_counts().head(10)
            
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, name=col,
                       marker_color=app_color_palette[i % len(app_color_palette)],
                       showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            height=550,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16)
        )
        
        fig.update_xaxes(gridcolor='rgba(139,92,246,0.2)', zerolinecolor='rgba(139,92,246,0.3)',
                         tickfont=dict(color='#8B5CF6', size=10),
                         title_font=dict(color='#7C3AED', size=11))
        fig.update_yaxes(gridcolor='rgba(139,92,246,0.2)', zerolinecolor='rgba(139,92,246,0.3)',
                         tickfont=dict(color='#8B5CF6', size=10),
                         title_font=dict(color='#7C3AED', size=11))
        
        fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/research/plots/categorical_distributions.html", 
                       include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# 5. Feature Correlations
print("\n=== Feature Correlations ===")
data_for_corr = train_data.copy()

# Encode target and categorical variables for correlation analysis
if train_data[target_col].dtype == 'object':
    # Try to infer binary encoding
    unique_vals = train_data[target_col].unique()
    if len(unique_vals) == 2:
        data_for_corr[target_col] = (train_data[target_col] == unique_vals[0]).astype(int)
else:
    data_for_corr[target_col] = train_data[target_col]

# Select numerical columns including encoded target
numerical_with_target = data_for_corr.select_dtypes(include=[np.number]).columns
correlation_matrix = data_for_corr[numerical_with_target].corr()

# Create correlation heatmap
fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='RdBu_r',
    zmid=0,
    text=np.round(correlation_matrix.values, 2),
    texttemplate="%{text}",
    textfont={"size": 10}
))

fig.update_layout(
    height=550,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=12),
    title_font=dict(color='#7C3AED', size=16),
    xaxis=dict(
        tickfont=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=11)
    ),
    yaxis=dict(
        tickfont=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=11)
    )
)

fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/research/plots/feature_correlations.html", 
               include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# Show correlations with target
if target_col in correlation_matrix.columns:
    target_correlations = correlation_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    print(f"\nCorrelations with target variable '{target_col}':")
    print(target_correlations)

# 6. Top numerical feature analysis by target
if numerical_cols and len(numerical_cols) > 0:
    print("\n=== Top Numerical Feature by Target ===")
    # Choose the first numerical feature for analysis
    feature_col = numerical_cols[0]
    
    fig = go.Figure()
    
    for i, target_value in enumerate(train_data[target_col].unique()):
        subset = train_data[train_data[target_col] == target_value][feature_col]
        fig.add_trace(go.Histogram(
            x=subset,
            name=f'{target_col}={target_value}',
            opacity=0.7,
            marker_color=app_color_palette[i]
        ))
    
    fig.update_layout(
        barmode='overlay',
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B5CF6', size=12),
        title_font=dict(color='#7C3AED', size=16),
        xaxis=dict(
            title=feature_col,
            gridcolor='rgba(139,92,246,0.2)',
            zerolinecolor='rgba(139,92,246,0.3)',
            tickfont=dict(color='#8B5CF6', size=11),
            title_font=dict(color='#7C3AED', size=12)
        ),
        yaxis=dict(
            title='Count',
            gridcolor='rgba(139,92,246,0.2)',
            zerolinecolor='rgba(139,92,246,0.3)',
            tickfont=dict(color='#8B5CF6', size=11),
            title_font=dict(color='#7C3AED', size=12)
        ),
        legend=dict(font=dict(color='#8B5CF6', size=11))
    )
    
    fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/research/plots/feature_by_target.html", 
                   include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# 7. Top categorical feature analysis
if categorical_cols and len(categorical_cols) > 0:
    print("\n=== Top Categorical Feature Analysis ===")
    cat_feature = categorical_cols[0]
    
    # Create crosstab
    crosstab = pd.crosstab(train_data[cat_feature], train_data[target_col], normalize='index') * 100
    
    fig = go.Figure()
    
    for i, target_value in enumerate(crosstab.columns):
        fig.add_trace(go.Bar(
            name=f'{target_col}={target_value}',
            x=crosstab.index,
            y=crosstab[target_value],
            marker_color=app_color_palette[i]
        ))
    
    fig.update_layout(
        barmode='stack',
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B5CF6', size=12),
        title_font=dict(color='#7C3AED', size=16),
        xaxis=dict(
            title=cat_feature,
            gridcolor='rgba(139,92,246,0.2)',
            zerolinecolor='rgba(139,92,246,0.3)',
            tickfont=dict(color='#8B5CF6', size=10),
            title_font=dict(color='#7C3AED', size=12),
            tickangle=45
        ),
        yaxis=dict(
            title='Percentage (%)',
            gridcolor='rgba(139,92,246,0.2)',
            zerolinecolor='rgba(139,92,246,0.3)',
            tickfont=dict(color='#8B5CF6', size=11),
            title_font=dict(color='#7C3AED', size=12)
        ),
        legend=dict(font=dict(color='#8B5CF6', size=11))
    )
    
    fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/research/plots/categorical_by_target.html", 
                   include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
    
    print(f"Distribution of {cat_feature} by target:")
    print(crosstab)

# 8. Outlier Analysis for numerical features
if numerical_cols and len(numerical_cols) > 0:
    print("\n=== Outlier Analysis ===")
    feature_col = numerical_cols[0]
    
    fig = go.Figure()
    
    for i, target_value in enumerate(train_data[target_col].unique()):
        subset = train_data[train_data[target_col] == target_value][feature_col]
        fig.add_trace(go.Box(
            y=subset,
            name=f'{target_col}={target_value}',
            marker_color=app_color_palette[i]
        ))
    
    fig.update_layout(
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B5CF6', size=12),
        title_font=dict(color='#7C3AED', size=16),
        xaxis=dict(
            title='Target Variable',
            gridcolor='rgba(139,92,246,0.2)',
            zerolinecolor='rgba(139,92,246,0.3)',
            tickfont=dict(color='#8B5CF6', size=11),
            title_font=dict(color='#7C3AED', size=12)
        ),
        yaxis=dict(
            title=feature_col,
            gridcolor='rgba(139,92,246,0.2)',
            zerolinecolor='rgba(139,92,246,0.3)',
            tickfont=dict(color='#8B5CF6', size=11),
            title_font=dict(color='#7C3AED', size=12)
        ),
        legend=dict(font=dict(color='#8B5CF6', size=11))
    )
    
    fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/research/plots/outlier_analysis.html", 
                   include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

print("\n=== EDA Complete ===")
print("All plots saved to /Users/yuvalheffetz/ds-agent-projects/session_313737e4-b92d-4cb9-8eb5-68f5df26d5d6/research/plots/")