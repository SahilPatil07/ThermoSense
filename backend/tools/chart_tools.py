# backend/tools/chart_tools.py
"""
Complete Chart System - Generation + Analysis
All chart functionality in one place for easy maintenance
"""
import uuid
import logging
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    
    # Professional styling
    # Professional styling
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')
    sns.set_context("paper", font_scale=1.3)
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChartGenerator:
    """All-in-one chart generation and analysis"""
    
    # Constants
    DEFAULT_DPI = 300
    DEFAULT_FIGSIZE = (12, 7)  # Standard publication size
    
    # Publication-Ready Color Palette (High Contrast, Colorblind Friendly)
    COLORS = [
        '#0077BB',  # Strong Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
        '#EE3377',  # Magenta
        '#BBBBBB',  # Grey
        '#000000'   # Black
    ]
    
    def __init__(self, llm_client=None, dpi: int = DEFAULT_DPI, figsize: tuple = DEFAULT_FIGSIZE):
        self.llm = llm_client
        self.dpi = dpi
        self.figsize = figsize
        
        # Set global style
        if PLOTTING_AVAILABLE:
            try:
                # Try to use a clean style
                plt.style.use('seaborn-v0_8-whitegrid')
            except:
                pass
            
            # Custom rcParams for publication quality
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                'font.size': 11,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'axes.titleweight': 'bold',
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16,
                'lines.linewidth': 2.0,
                'grid.alpha': 0.3
            })
    
    def generate_chart(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> Tuple[bool, str, str, str, str, Dict]:
        """
        Generate chart + AI analysis in ONE function
        
        Returns:
            (success, message, chart_path, chart_id, summary, statistics)
        """
        if not PLOTTING_AVAILABLE:
            return False, "Plotting libraries (matplotlib/seaborn) not installed", "", "", "", {}

        try:
            x_col = params.get('x_column')
            y_cols = params.get('y_columns', [])
            chart_type = params.get('chart_type', 'line')
            title = params.get('title', 'Chart')
            
            # Validate
            if not x_col or not y_cols:
                return False, "Missing columns", "", "", "", {}
            
            # Prepare data
            # We select only the columns we need and remove any empty rows.
            df_plot = df[[x_col] + y_cols].dropna().copy()
            
            if df_plot.empty:
                return False, "No valid data", "", "", "", {}
            
            # Note: We are NOT sorting the data. We plot it exactly as it appears in the file.
            # This ensures the chart represents the true sequence of the data.
            # Generate chart ID
            chart_id = uuid.uuid4().hex[:12]
            
            # Create chart
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Plot
            for idx, y_col in enumerate(y_cols):
                color = self.COLORS[idx % len(self.COLORS)]
                
                if chart_type == 'scatter':
                    ax.scatter(df_plot[x_col].values, df_plot[y_col].values, label=y_col, 
                              alpha=0.6, s=30, color=color, edgecolors='white')
                elif chart_type == 'bar':
                    # Smart Aggregation for large datasets
                    MAX_BARS = 50
                    plot_df_agg = df_plot.copy()
                    
                    if len(plot_df_agg) > MAX_BARS:
                        # Check if X is numeric/datetime or categorical
                        is_numeric_x = pd.api.types.is_numeric_dtype(plot_df_agg[x_col]) or pd.api.types.is_datetime64_any_dtype(plot_df_agg[x_col])
                        
                        if is_numeric_x:
                            try:
                                if pd.api.types.is_datetime64_any_dtype(plot_df_agg[x_col]):
                                    plot_df_agg['bin'] = pd.cut(plot_df_agg[x_col], bins=MAX_BARS)
                                    plot_df_agg[x_col] = plot_df_agg['bin'].apply(lambda x: x.mid)
                                else:
                                    plot_df_agg['bin'] = pd.cut(plot_df_agg[x_col], bins=MAX_BARS)
                                    plot_df_agg[x_col] = plot_df_agg['bin'].apply(lambda x: x.mid)
                                
                                agg_dict = {col: 'mean' for col in y_cols}
                                plot_df_agg = plot_df_agg.groupby(x_col, as_index=False).agg(agg_dict)
                            except:
                                plot_df_agg = plot_df_agg.head(MAX_BARS)
                        else:
                            # Categorical
                            first_y = y_cols[0]
                            plot_df_agg = plot_df_agg.sort_values(by=first_y, ascending=False)
                            if len(plot_df_agg) > MAX_BARS:
                                top_df = plot_df_agg.iloc[:MAX_BARS-1]
                                other_df = plot_df_agg.iloc[MAX_BARS-1:]
                                other_row = {x_col: 'Other'}
                                for col in y_cols:
                                    other_row[col] = other_df[col].mean()
                                plot_df_agg = pd.concat([top_df, pd.DataFrame([other_row])], ignore_index=True)
                    
                    # Plotting
                    width = 0.8 / len(y_cols)
                    x_indices = np.arange(len(plot_df_agg))
                    
                    for i, y_col in enumerate(y_cols):
                        ax.bar(x_indices + i * width, plot_df_agg[y_col].values, width=width, label=y_col,
                               color=self.COLORS[i % len(self.COLORS)], alpha=0.8)
                    
                    # Set x-ticks
                    ax.set_xticks(x_indices + width * (len(y_cols) - 1) / 2)
                    ax.set_xticklabels(plot_df_agg[x_col].values, rotation=45, ha='right')

                elif chart_type == 'histogram':
                    ax.hist(df_plot[y_col].values, bins=20, label=y_col, alpha=0.6, color=color, edgecolor='white')
                    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                
                elif chart_type == 'box':
                    # Box plot usually ignores X-axis for distribution, or groups by X
                    # Here we just show distribution of Y columns
                    # We'll do a single boxplot for all Y columns at once later, but loop structure makes it tricky
                    # So we collect data first? No, let's just do it simple:
                    # If box, we might ignore X or treat X as category. 
                    # Let's assume simple distribution of Y for now.
                    pass # Handled outside loop for better layout
                
                elif chart_type == 'area':
                    ax.fill_between(df_plot[x_col].values, df_plot[y_col].values, label=y_col,
                                   color=color, alpha=0.4)
                    ax.plot(df_plot[x_col].values, df_plot[y_col].values, color=color, linewidth=1.5)

                else: # Default Line Chart
                    # Enhanced line plot with better rendering
                    # Handle case where all values might be identical
                    y_data = df_plot[y_col].values
                    x_data = df_plot[x_col].values
                    
                    # Check for constant data
                    if np.ptp(y_data) == 0:  # All values identical
                        # Still plot as line, but add visual indicator
                        ax.axhline(y=y_data[0], color=color, linewidth=2.5, 
                                  label=f'{y_col} (constant: {y_data[0]:.2f})',
                                  linestyle='--', alpha=0.7)
                    else:
                        # Normal line plot with ALL data points
                        line, = ax.plot(x_data, y_data, label=y_col,
                               linewidth=2.0, alpha=0.9, color=color)
                               
                        # Add Max/Min Annotations
                        try:
                            max_idx = np.argmax(y_data)
                            min_idx = np.argmin(y_data)
                            
                            # Annotate Max
                            ax.annotate(f'Max: {y_data[max_idx]:.2f}',
                                       xy=(x_data[max_idx], y_data[max_idx]),
                                       xytext=(10, 10), textcoords='offset points',
                                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black'),
                                       fontsize=8, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                                       
                            # Annotate Min
                            ax.annotate(f'Min: {y_data[min_idx]:.2f}',
                                       xy=(x_data[min_idx], y_data[min_idx]),
                                       xytext=(10, -20), textcoords='offset points',
                                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.2', color='black'),
                                       fontsize=8, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                        except:
                            pass # Skip if annotation fails
            
            # Special handling for Box Plot (outside loop)
            if chart_type == 'box':
                data_to_plot = [df_plot[col].values for col in y_cols]
                ax.boxplot(data_to_plot, labels=y_cols, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', color='blue'),
                          medianprops=dict(color='red'))
                ax.set_xlabel('Variables', fontsize=12, fontweight='bold')

            
            # Professional Styling - Industry Standard
            ax.set_xlabel(str(x_col), fontsize=13, fontweight='600', labelpad=12)
            ax.set_ylabel('Value', fontsize=13, fontweight='600', labelpad=12)
            ax.set_title(title, fontsize=15, fontweight='700', pad=18)
            
            # Enhanced Legend with better positioning
            if len(y_cols) > 0:
                ax.legend(loc='best', frameon=True, fancybox=True,
                         shadow=True, fontsize=10, edgecolor='#999999',
                         framealpha=0.95, borderpad=1)
            
            # Professional Grid System
            ax.grid(True, which='major', color='#d0d0d0', linestyle='-', linewidth=0.8, alpha=0.7)
            ax.grid(True, which='minor', color='#e8e8e8', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.minorticks_on()
            
            # Clean spine styling
            for spine in ax.spines.values():
                spine.set_edgecolor('#666666')
                spine.set_linewidth(1.0)
            
            # Optimize tick labels
            ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1)
            ax.tick_params(axis='both', which='minor', labelsize=8, length=3, width=0.5)
            plt.xticks(rotation=45, ha='right')
            
            # Tight layout for optimal space usage
            plt.tight_layout(pad=1.5)
            
            # Save
            chart_path = f"{output_path}_{chart_id}.png"
            plt.savefig(chart_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            # Generate analysis
            user_query = params.get('user_query')
            summary, stats = self._analyze_data(df_plot, x_col, y_cols, chart_type, user_query)
            
            logger.info(f"✅ Chart + analysis: {chart_id}")
            
            return True, "Success", chart_path, chart_id, summary, stats
        
        except Exception as e:
            logger.exception("Chart generation failed")
            return False, str(e), "", "", "", {}
    
    def _analyze_data(
        self, 
        df: pd.DataFrame, 
        x_col: str, 
        y_cols: list,
        chart_type: str,
        user_query: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """Generate AI summary + statistics"""
        
        # Compute stats
        stats = {"total_points": len(df), "columns": {}}
        
        for col in y_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_data = df[col].dropna()
                
                # Trend detection
                slope = np.polyfit(np.arange(len(col_data)), col_data.values, 1)[0]
                if abs(slope) < 0.01:
                    trend = "stable"
                elif slope > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"
                
                stats["columns"][col] = {
                    "mean": float(col_data.mean()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "std": float(col_data.std()),
                    "trend": trend
                }
        
        # Generate summary
        if self.llm:
            summary = self._llm_summary(x_col, y_cols, chart_type, stats, user_query)
        else:
            summary = self._simple_summary(x_col, y_cols, stats)
        
        return summary, stats
    
    def _llm_summary(self, x_col: str, y_cols: list, chart_type: str, stats: Dict, user_query: Optional[str] = None) -> str:
        """AI-powered summary with Chain-of-Thought"""
        
        stats_text = "\n".join([
            f"- {col}: mean={data['mean']:.2f}, range=[{data['min']:.2f}, {data['max']:.2f}], trend={data['trend']}"
            for col, data in stats["columns"].items()
        ])
        
        context_prompt = ""
        if user_query:
            context_prompt = f"\n**User Question/Context:**\nThe user asked: '{user_query}'\nFocus your analysis specifically on answering this question or addressing this context.\n"

        prompt = f"""As a Senior Thermal Engineer, analyze this data plot.
        
**Data Context:**
- X-axis: {x_col}
- Y-axis: {', '.join(y_cols)}
- Chart Type: {chart_type}
- Data Points: {stats['total_points']}
{context_prompt}
**Statistical Summary:**
{stats_text}

**Analysis Requirements:**
1. **Trend Analysis**: Describe the behavior (linear, exponential, fluctuating, stable).
2. **Critical Points**: Identify max/min values and any sudden changes or anomalies.
3. **Engineering Insight**: Relate these values to potential physical phenomena (e.g., thermal saturation, cooling efficiency, sensor noise, rapid heating).
4. **Actionable Conclusion**: Is this behavior expected? Does it indicate a problem?

**Output Format:**
Provide a concise, professional engineering summary (approx. 4-5 sentences). Do not use bullet points. Focus on technical interpretation.

**Analysis:**"""

        try:
            response = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except:
            return self._simple_summary(x_col, y_cols, stats)
    
    def _simple_summary(self, x_col: str, y_cols: list, stats: Dict) -> str:
        """Fallback summary without LLM"""
        
        parts = [f"Chart shows {', '.join(y_cols)} vs {x_col} with {stats['total_points']} data points."]
        
        for col, data in stats["columns"].items():
            trend_word = {"increasing": "increases", "decreasing": "decreases", "stable": "remains stable"}.get(data["trend"], "varies")
            parts.append(f"{col} {trend_word}, ranging from {data['min']:.2f} to {data['max']:.2f} (mean: {data['mean']:.2f}).")
        
        parts.append("This data provides insights into system thermal behavior.")
        
        return " ".join(parts)

# Singleton
_chart_generator = None

def get_chart_generator(llm_client=None, dpi: int = 300, figsize: tuple = (16, 9)):
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = ChartGenerator(llm_client, dpi, figsize)
    return _chart_generator
