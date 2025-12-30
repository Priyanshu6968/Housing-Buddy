"""
Exploratory Data Analysis Script

Generates visualizations for the property valuation dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_EDA = PROJECT_ROOT / "reports" / "eda"


def load_processed_data():
    """Load processed CSV files."""
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def plot_price_distribution(df, save_path):
    """Plot the distribution of property prices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Price ($)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
    axes[0].axvline(df['price'].mean(), color='red', linestyle='--', label=f'Mean: ${df["price"].mean():,.0f}')
    axes[0].axvline(df['price'].median(), color='green', linestyle='--', label=f'Median: ${df["price"].median():,.0f}')
    axes[0].legend()
    
    # Log-transformed histogram
    axes[1].hist(np.log1p(df['price']), bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Log(Price + 1)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Log-Transformed Price Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_heatmap(df, save_path):
    """Plot correlation heatmap of numeric features."""
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, ax=ax,
                square=True, linewidths=0.5,
                annot_kws={'size': 8})
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_price_vs_sqft(df, save_path):
    """Plot price vs sqft_living relationship."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Find sqft column
    sqft_col = None
    for col in df.columns:
        if 'sqft' in col.lower() and 'living' in col.lower():
            sqft_col = col
            break
    
    if sqft_col is None:
        # Try to find any sqft column
        for col in df.columns:
            if 'sqft' in col.lower():
                sqft_col = col
                break
    
    if sqft_col:
        ax.scatter(df[sqft_col], df['price'], alpha=0.5, s=10)
        ax.set_xlabel(f'{sqft_col}', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f'Price vs {sqft_col}', fontsize=14, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(df[sqft_col], df['price'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[sqft_col].min(), df[sqft_col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend line')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'sqft_living column not found', 
                transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_price_vs_bedrooms(df, save_path):
    """Plot price vs bedrooms box plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find bedrooms column
    bedroom_col = None
    for col in df.columns:
        if 'bedroom' in col.lower():
            bedroom_col = col
            break
    
    if bedroom_col:
        # Limit to reasonable bedroom counts
        df_filtered = df[df[bedroom_col] <= 10].copy()
        
        sns.boxplot(x=bedroom_col, y='price', data=df_filtered, ax=ax)
        ax.set_xlabel('Number of Bedrooms', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('Price Distribution by Number of Bedrooms', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'bedrooms column not found', 
                transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_geographic_distribution(df, save_path):
    """Plot geographic distribution of properties."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Check for lat/long columns
    if 'lat' in df.columns and 'long' in df.columns:
        # Scatter plot colored by price
        scatter = axes[0].scatter(df['long'], df['lat'], 
                                   c=df['price'], cmap='viridis',
                                   alpha=0.5, s=5)
        axes[0].set_xlabel('Longitude', fontsize=12)
        axes[0].set_ylabel('Latitude', fontsize=12)
        axes[0].set_title('Property Locations (colored by price)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=axes[0], label='Price ($)')
        
        # Density plot
        sns.kdeplot(x=df['long'], y=df['lat'], ax=axes[1], 
                    cmap='Reds', fill=True, levels=20)
        axes[1].set_xlabel('Longitude', fontsize=12)
        axes[1].set_ylabel('Latitude', fontsize=12)
        axes[1].set_title('Property Density', fontsize=14, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'lat/long columns not found', 
                     transform=axes[0].transAxes, ha='center')
        axes[1].text(0.5, 0.5, 'lat/long columns not found', 
                     transform=axes[1].transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance_correlation(df, save_path):
    """Plot feature correlation with price."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if 'price' in numeric_df.columns:
        # Calculate correlation with price
        correlations = numeric_df.corr()['price'].drop('price').sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['green' if x > 0 else 'red' for x in correlations]
        correlations.plot(kind='barh', color=colors, ax=ax)
        
        ax.set_xlabel('Correlation with Price', fontsize=12)
        ax.set_title('Feature Correlations with Price', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def generate_summary_report(df, save_path):
    """Generate a text summary of the dataset."""
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DATASET SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Total features: {len(df.columns)}\n\n")
        
        f.write("PRICE STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean:   ${df['price'].mean():,.2f}\n")
        f.write(f"  Median: ${df['price'].median():,.2f}\n")
        f.write(f"  Std:    ${df['price'].std():,.2f}\n")
        f.write(f"  Min:    ${df['price'].min():,.2f}\n")
        f.write(f"  Max:    ${df['price'].max():,.2f}\n\n")
        
        f.write("MISSING VALUES:\n")
        f.write("-" * 40 + "\n")
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                f.write(f"  {col}: {count}\n")
        if missing.sum() == 0:
            f.write("  No missing values\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Saved: {save_path}")


def run_eda():
    """Run complete EDA and save all visualizations."""
    # Ensure output directory exists
    REPORTS_EDA.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading processed data...")
    train_df, test_df = load_processed_data()
    
    print(f"\nTraining data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    print(f"\nColumns: {list(train_df.columns)}")
    
    # Generate visualizations
    print("\nGenerating EDA visualizations...")
    
    print("\n1. Price Distribution...")
    plot_price_distribution(train_df, REPORTS_EDA / "price_distribution.png")
    
    print("2. Correlation Heatmap...")
    plot_correlation_heatmap(train_df, REPORTS_EDA / "correlation_heatmap.png")
    
    print("3. Price vs Sqft Living...")
    plot_price_vs_sqft(train_df, REPORTS_EDA / "price_vs_sqft.png")
    
    print("4. Price vs Bedrooms...")
    plot_price_vs_bedrooms(train_df, REPORTS_EDA / "price_vs_bedrooms.png")
    
    print("5. Geographic Distribution...")
    plot_geographic_distribution(train_df, REPORTS_EDA / "geographic_distribution.png")
    
    print("6. Feature Correlations with Price...")
    plot_feature_importance_correlation(train_df, REPORTS_EDA / "feature_correlations.png")
    
    print("7. Summary Report...")
    generate_summary_report(train_df, REPORTS_EDA / "summary_report.txt")
    
    print("\n" + "=" * 50)
    print("EDA Complete!")
    print(f"All visualizations saved to: {REPORTS_EDA}")
    print("=" * 50)


if __name__ == "__main__":
    run_eda()
