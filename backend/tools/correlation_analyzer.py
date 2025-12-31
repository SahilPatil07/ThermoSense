"""
Correlation Analyzer for Automotive Thermal Data
Generates heatmaps with Pearson & Spearman correlation
Provides industry-specific insights
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional
from scipy import stats


class CorrelationAnalyzer:
    """
    Correlation analysis with automotive thermal insights
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.dpi = 300
        self.figsize = (12, 10)
    
    def generate_heatmap(self, df: pd.DataFrame, method: str, output_path: str) -> Tuple[bool, str, str]:
        """
        Generate correlation heatmap
        
        Args:
            df: DataFrame with numeric data
            method: 'pearson' or 'spearman'
            output_path: Save location
        
        Returns:
            (success, analysis_text, path)
        """
        try:
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[np.number]).copy()

            # If there's a common index-like "ID" column, drop it to avoid artificial correlations
            if 'ID' in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=['ID'])

            if numeric_df.shape[1] < 2:
                return False, "Not enough numeric columns for correlation analysis", ""
            
            # Guard against constant columns (std == 0) which make correlation undefined
            const_cols = [col for col in numeric_df.columns if np.isclose(numeric_df[col].std(ddof=0), 0.0)]
            if const_cols:
                print(f"Dropping constant columns: {const_cols}")
                numeric_df = numeric_df.drop(columns=const_cols)
            
            if numeric_df.shape[1] < 2:
                return False, "Not enough non-constant numeric columns for correlation analysis. Please select columns with varying data.", ""
            
            # Calculate correlation
            method_lower = method.lower()
            if method_lower == 'spearman':
                corr_matrix = numeric_df.corr(method='spearman')
                title = "Spearman Correlation Heatmap (Rank-Based)"
            else:
                corr_matrix = numeric_df.corr(method='pearson')
                title = "Pearson Correlation Heatmap (Linear)"
            
            # Replace any remaining NaNs with 0
            corr_matrix = corr_matrix.fillna(0.0)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            mask = np.triu(np.ones(corr_matrix.shape, dtype=bool))
            
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.8,
                cbar_kws={"shrink": 0.75, "label": "Correlation Coefficient"},
                ax=ax,
                vmin=-1,
                vmax=1
            )
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            
            # Save
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # Generate analysis
            analysis = self._generate_correlation_analysis(corr_matrix, method, numeric_df.columns.tolist())
            
            print(f"Heatmap saved: {output_path}")
            return True, analysis, output_path
        
        except Exception as e:
            print(f"Heatmap error: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e), ""
    
    def _generate_correlation_analysis(self, corr_matrix: pd.DataFrame, method: str, columns: list) -> str:
        """
        Generate intelligent analysis of correlations using LLM if available
        """
        # 1. Try LLM Analysis
        if self.llm:
            try:
                # Create a more structured prompt to avoid hallucinations
                top_corr_str = corr_matrix.unstack().sort_values(key=abs, ascending=False).drop_duplicates().head(10).to_string()
                
                prompt = f"""
                You are an expert automotive data scientist. Analyze this correlation matrix for a thermal dataset.
                
                Method: {method}
                Columns: {', '.join(columns)}
                
                Top Correlations (r-value):
                {top_corr_str}
                
                Instructions:
                1. Identify the most significant 2-3 positive and 2-3 negative correlations.
                2. Provide PHYSICAL interpretations relevant to automotive thermal systems (e.g. "Higher RPM increases Coolant Temp due to friction/combustion heat").
                3. If correlations are nonsense (e.g. Time vs ID), ignore them.
                4. Keep it under 150 words. Use bullet points.
                
                Output format:
                **Statistical Insights:**
                - [Point 1]
                - [Point 2]
                
                **Engineering Interpretation:**
                - [Interpretation 1]
                - [Interpretation 2]
                """
                
                response = self.llm.get_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    system_message="You are a helpful automotive engineering assistant."
                )
                
                if response:
                    return response
            except Exception as e:
                print(f"LLM Analysis failed: {e}. Falling back to rule-based analysis.")

        # 2. Fallback Rule-Based Analysis
        strong_positive = []
        strong_negative = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if pd.isna(corr_val): continue
                
                if abs(corr_val) > 0.7:
                    pair = (corr_matrix.columns[i], corr_matrix.columns[j], float(corr_val))
                    if corr_val > 0:
                        strong_positive.append(pair)
                    else:
                        strong_negative.append(pair)
        
        method_name = "Spearman (rank-based)" if method.lower() == 'spearman' else "Pearson (linear)"
        analysis = f"**{method_name} Correlation Analysis (Automated)**\n\n"
        
        if strong_positive:
            analysis += "**Strong Positive Correlations:**\n"
            for p1, p2, r in sorted(strong_positive, key=lambda x: abs(x[2]), reverse=True)[:3]:
                analysis += f"- {p1} <-> {p2}: {r:.2f}\n"
        
        if strong_negative:
            analysis += "\n**Strong Negative Correlations:**\n"
            for p1, p2, r in sorted(strong_negative, key=lambda x: abs(x[2]), reverse=True)[:3]:
                analysis += f"- {p1} <-> {p2}: {r:.2f}\n"
                
        if not strong_positive and not strong_negative:
            analysis += "No strong correlations (|r| > 0.7) detected.\n"
            
        return analysis

    def calculate_correlation_stats(self, df: pd.DataFrame, col1: str, col2: str) -> Dict:
        """
        Calculate detailed correlation statistics
        """
        try:
            data1 = pd.to_numeric(df[col1], errors='coerce').dropna()
            data2 = pd.to_numeric(df[col2], errors='coerce').dropna()
            
            # Align data
            common_idx = data1.index.intersection(data2.index)
            data1 = data1.loc[common_idx]
            data2 = data2.loc[common_idx]
            
            if len(data1) < 3:
                return {}
            
            # If either variable is constant, pearsonr will error; check variances
            if np.isclose(data1.std(ddof=0), 0.0) or np.isclose(data2.std(ddof=0), 0.0):
                return {
                    "pearson_r": float('nan'), "pearson_p": float('nan'),
                    "spearman_r": float('nan'), "spearman_p": float('nan'),
                    "sample_size": len(data1), "significant": False
                }
            
            pearson_r, pearson_p = stats.pearsonr(data1, data2)
            spearman_r, spearman_p = stats.spearmanr(data1, data2)
            
            return {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "sample_size": len(data1),
                "significant": pearson_p < 0.05
            }
        
        except Exception as e:
            print(f"Correlation stats error: {e}")
            return {}
