import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from load_data import load_data
from generate_plots import (
    numeric_statistics, 
    categorical_statistics, 
    generate_boxplot, 
    generate_violinplot, 
    generate_regression_plot, 
    generate_correlation_heatmap, 
    generate_conditional_histograms
)

def main():
    
    output_dir = "results"
    
    os.makedirs(output_dir, exist_ok=True)

    df = load_data('data/obesity_data.csv')

    
    
    print("Obliczanie statystyk numerycznych...")
    numeric_stats = numeric_statistics(df)
    numeric_stats_path = os.path.join(output_dir, "numeric_statistics.csv")
    numeric_stats.to_csv(numeric_stats_path)
    print(f"Zapisano: {numeric_stats_path}")

    print("Obliczanie statystyk kategorycznych...")
    categorical_stats = categorical_statistics(df)
    categorical_stats_path = os.path.join(output_dir, "categorical_statistics.json")
    categorical_stats.to_json(categorical_stats_path, orient="columns", indent=4)
    print(f"Zapisano: {categorical_stats_path}")

    
    
    print("\nRozpoczynam generowanie wykresów...")

    print("Generowanie: Boxplot (Wiek vs Poziom otyłości)")
    generate_boxplot(df, x_feature='NObeyesdad', y_feature='Age', output_dir=output_dir)

    print("Generowanie: Violinplot (Waga vs Płeć)")
    generate_violinplot(df, x_feature='Gender', y_feature='Weight', output_dir=output_dir)

    print("Generowanie: Histogramy warunkowe (wg NObeyesdad)")
    generate_conditional_histograms(df, hue='NObeyesdad', output_dir=output_dir)

    print("Generowanie: Heatmapa korelacji")
    generate_correlation_heatmap(df, output_dir=output_dir)
    
    print("Generowanie: Wykres regresji (Waga vs Wzrost)")
    generate_regression_plot(df, x_feature='Weight', y_feature='Height', output_dir=output_dir)
    
    print(f"\nGotowe! Wszystkie wykresy i statystyki zapisano w folderze '{output_dir}'.")


if __name__ == "__main__":
    main()