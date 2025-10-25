import pandas as pd
from load_data import load_data
import seaborn as sns
import matplotlib.pyplot as plt
import os

def numeric_statistics(df):

    stats = {}

    for column in df.select_dtypes(include=['number']).columns:
        mean_value = df[column].mean()                
        median_value = df[column].median()            
        min_value = df[column].min()                   
        max_value = df[column].max()                 
        std_value = df[column].std()                 
        percentile_5 = df[column].quantile(0.05)       
        percentile_95 = df[column].quantile(0.95)       
        missing_values = df[column].isnull().sum()   
        
        stats[column] = {
            'mean': mean_value,
            'median': median_value,
            'min': min_value,
            'max': max_value,
            'std': std_value,
            '5th_percentile': percentile_5,
            '95th_percentile': percentile_95,
            'missing_values': missing_values
        }

    return pd.DataFrame(stats)


def categorical_statistics(df):

    stats = {}

    for column in df.select_dtypes(include=['object']).columns:
        unique_classes = df[column].nunique()
        missing_values = df[column].isnull().sum()
        class_proportions = df[column].value_counts(normalize=True)

        stats[column] = {
            'unique_classes': unique_classes,
            'missing_values': missing_values,
            'class_proportions': class_proportions.to_dict()
        }

    return pd.DataFrame(stats)

def generate_boxplot(df, x_feature, y_feature, output_dir="results"):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df[x_feature], y=df[y_feature], palette='pastel')
    plt.title(f'Boxplot of {y_feature} by {x_feature}', fontsize=14, fontweight='bold')
    plt.xlabel(x_feature, fontsize=12)
    plt.ylabel(y_feature, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plot_path = os.path.join(output_dir, f"boxplot_{x_feature}_vs_{y_feature}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Zapisano: {plot_path}")


def generate_violinplot(df, x_feature, y_feature, output_dir="results"):
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=df[x_feature], y=df[y_feature], palette='pastel', inner="quartile")
    plt.title(f'Violinplot of {y_feature} by {x_feature}', fontsize=14, fontweight='bold')
    plt.xlabel(x_feature, fontsize=12)
    plt.ylabel(y_feature, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plot_path = os.path.join(output_dir, f"violinplot_{x_feature}_vs_{y_feature}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Zapisano: {plot_path}")


def generate_regression_plot(df, x_feature, y_feature, output_dir="results"):
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df[x_feature], y=df[y_feature], scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    plt.xlabel(x_feature, fontsize=12)
    plt.ylabel(y_feature, fontsize=12)
    plt.title(f"Linear Regression: {x_feature} vs {y_feature}", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = os.path.join(output_dir, f"regression_{x_feature}_vs_{y_feature}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Zapisano: {plot_path}")



def generate_correlation_heatmap(df, output_dir="results"):
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plot_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Zapisano: {plot_path}")


def generate_conditional_histograms(df, hue, output_dir="."):
    
    if hue not in df.columns:
        raise ValueError(f"Zmiennej {hue} nie znaleziono w zbiorze danych.")
    
    numeric_features = df.select_dtypes(include=['number']).columns
    
    for feature in numeric_features:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=feature, hue=hue, multiple="dodge", bins=20, palette="Set1", alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Conditional Histogram of {feature} by {hue}")
        
        plot_path = os.path.join(output_dir, f"conditional_histogram_{feature}_by_{hue}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Zapisano: {plot_path}")



def generate_regression_plots(df, x_feature, y_feature, output_dir="results"):

    if x_feature not in df.columns or y_feature not in df.columns:
        raise ValueError(f"Jedna z podanych cech ({x_feature}, {y_feature}) nie istnieje w zbiorze danych.")
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df[x_feature], y=df[y_feature], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    plt.xlabel(x_feature, fontsize=12)
    plt.ylabel(y_feature, fontsize=12)
    plt.title(f"Linear Regression: {x_feature} vs {y_feature}", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join(output_dir, f"regression_{x_feature}_vs_{y_feature}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Zapisano: {plot_path}")