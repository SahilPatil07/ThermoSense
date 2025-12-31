# backend/tools/chart_tools_plotly.py
"""
Plotly-Based Interactive Chart System
Provides interactive charts with zoom, pan, range selection, and multi-format downloads
"""
import uuid
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json
import threading
import time
from backend.tools.json_utils import safe_json_dumps
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PlotlyChartGenerator:
    """Interactive chart generation with plotly"""
    
    # Constants
    DEFAULT_WIDTH = 1200
    DEFAULT_HEIGHT = 700
    
    # Professional Color Palette (matching matplotlib version)
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
    
    def __init__(self, llm_client=None, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.llm = llm_client
        self.width = width
        self.height = height
    
    def generate_chart(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> Tuple[bool, str, str, str, str, str, Dict, Optional[str]]:
        """
        Generate interactive plotly chart + AI analysis
        
        Returns:
            (success, message, html_path, png_path, chart_id, summary, statistics, plotly_json)
        """
        try:
            x_col = params.get('x_column')
            y_cols = params.get('y_columns', [])
            chart_type = params.get('chart_type', 'line')
            title = params.get('title', 'Interactive Chart')
            
            # Validate
            if not x_col or not y_cols:
                return False, "Missing columns", "", "", "", "", {}, None
            
            # Prepare data - only drop rows where X is missing
            df_plot = df.copy()
            
            # Ensure columns are unique (just in case)
            df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]
            
            if x_col in df_plot.columns:
                df_plot = df_plot.dropna(subset=[x_col])
            
            # CRITICAL: Sort by X-axis to ensure line charts are drawn correctly
            try:
                # Try numeric sort first
                df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors='coerce')
                df_plot = df_plot.sort_values(by=x_col)
            except Exception as e:
                logger.warning(f"Sorting failed: {e}")
                # Fallback to string sort or no sort
                try:
                    df_plot = df_plot.sort_values(by=x_col)
                except:
                    pass
            
            if df_plot.empty:
                return False, "No valid data", "", "", "", "", {}, None
            
            # Generate chart ID
            chart_id = uuid.uuid4().hex[:12]
            
            # Create plotly figure based on chart type
            fig = self._create_figure(df_plot, x_col, y_cols, chart_type, title)
            
            # Add range selector for line/scatter charts
            if chart_type in ['line', 'scatter', 'area']:
                self._add_range_selector(fig)
            
            # Configure layout for professional look
            self._configure_layout(fig, title)
            
            # Save as HTML (interactive)
            html_path = f"{output_path}_{chart_id}.html"
            fig.write_html(
                html_path,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'responsive': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'chart_{chart_id}',
                        'height': self.height,
                        'width': self.width,
                        'scale': 2
                    }
                }
            )
            
            # Save as PNG (for reports)
            png_path = f"{output_path}_{chart_id}.png"
            export_success = False
            
            try:
                # Use a thread to handle kaleido export with a timeout
                # This prevents hanging on Windows systems where kaleido can be unstable
                import plotly.io as pio
                
                def export_png():
                    nonlocal export_success
                    try:
                        pio.write_image(fig, png_path, format='png', width=self.width, height=self.height, scale=2)
                        export_success = True
                    except Exception as e:
                        logger.warning(f"Kaleido export thread failed: {e}")

                export_thread = threading.Thread(target=export_png)
                export_thread.daemon = True
                export_thread.start()
                export_thread.join(timeout=5.0) # 5 second timeout for PNG export
                
                if export_thread.is_alive():
                    logger.warning("Kaleido export timed out after 5s - falling back to Matplotlib")
                elif not export_success:
                    logger.warning("Kaleido export failed - falling back to Matplotlib")
                else:
                    logger.info(f"Kaleido export successful: {png_path}")
            
            except Exception as e:
                logger.warning(f"Kaleido setup failed: {e}")

            # Fallback to Matplotlib if kaleido failed or timed out
            if not export_success or not Path(png_path).exists():
                try:
                    logger.info("Generating static fallback PNG using Matplotlib...")
                    self._generate_static_fallback(df_plot, x_col, y_cols, chart_type, title, png_path)
                    if Path(png_path).exists():
                        logger.info(f"Matplotlib fallback successful: {png_path}")
                        export_success = True
                except Exception as e:
                    logger.error(f"Matplotlib fallback failed: {e}")
                    png_path = ""
            
            # Generate analysis
            user_query = params.get('user_query')
            summary, stats = self._analyze_data(df_plot, x_col, y_cols, chart_type, user_query)
            
            logger.info(f"Plotly chart + analysis: {chart_id}")
            
            # Prepare responsive JSON (remove hardcoded width/height)
            fig_json = fig.to_dict()
            if 'layout' in fig_json:
                fig_json['layout'].pop('width', None)
                fig_json['layout'].pop('height', None)
                fig_json['layout']['autosize'] = True
            
            return True, "Success", html_path, png_path, chart_id, summary, stats, safe_json_dumps(fig_json)
        
        except Exception as e:
            logger.exception("Chart generation failed")
            return False, str(e), "", "", "", "", {}, None

    def generate_heatmap(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> Tuple[bool, str, str, str, str, str, Dict, Optional[str]]:
        """
        Generate interactive heatmap (correlation or data)
        """
        try:
            # Generate chart ID
            chart_id = uuid.uuid4().hex[:12]
            
            # Calculate correlation matrix if not already a matrix
            if len(df.columns) > 1 and len(df) > 1:
                # Try to convert to numeric, dropping non-numeric columns
                df_numeric = df.select_dtypes(include=['number'])
                
                if df_numeric.empty:
                    return False, "No numeric data for heatmap", "", "", "", "", {}, None
                
                # For very large datasets, limit to most correlated columns
                num_cols = len(df_numeric.columns)
                if num_cols > 100:
                    logger.info(f"Heatmap: Reducing {num_cols} columns to top 100 most correlated")
                    # Calculate correlation and find top correlated columns
                    corr_matrix_full = df_numeric.corr()
                    # Get average absolute correlation for each column
                    avg_corr = corr_matrix_full.abs().mean().sort_values(ascending=False)
                    # Keep top 100 columns
                    top_cols = avg_corr.head(100).index.tolist()
                    df_numeric = df_numeric[top_cols]
                    num_cols = 100
                
                # Calculate correlation
                corr_matrix = df_numeric.corr()
                
                title = f"Correlation Heatmap ({num_cols} variables)"
                
                # Disable text annotations for large heatmaps (unreadable)
                show_text = num_cols <= 50
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f' if show_text else False,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title=title,
                    labels=dict(color="Correlation")
                )
            else:
                return False, "Insufficient data for heatmap", "", "", "", "", {}, None
            
            # Configure layout
            self._configure_layout(fig, title)
            
            # Save as HTML
            html_path = f"{output_path}_{chart_id}.html"
            fig.write_html(
                html_path,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'responsive': True
                }
            )
            
            # Save as PNG (Kaleido with timeout)
            png_path = f"{output_path}_{chart_id}.png"
            export_success = False
            
            try:
                import plotly.io as pio
                
                def export_png():
                    nonlocal export_success
                    try:
                        pio.write_image(fig, png_path, format='png', width=self.width, height=self.height, scale=2)
                        export_success = True
                    except Exception as e:
                        logger.warning(f"Kaleido export thread failed: {e}")

                export_thread = threading.Thread(target=export_png)
                export_thread.daemon = True
                export_thread.start()
                export_thread.join(timeout=5.0)
                
                if export_thread.is_alive():
                    logger.warning("Kaleido export timed out - falling back to Matplotlib")
                elif not export_success:
                    logger.warning("Kaleido export failed - falling back to Matplotlib")
                else:
                    logger.info(f"Kaleido export successful: {png_path}")
            
            except Exception as e:
                logger.warning(f"Kaleido setup failed: {e}")
            
            # Fallback to Matplotlib
            if not export_success or not Path(png_path).exists():
                try:
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt='.2f')
                    plt.title(title)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=100)
                    plt.close()
                    export_success = True
                    logger.info(f"Matplotlib fallback successful: {png_path}")
                except Exception as e:
                    logger.error(f"Matplotlib fallback failed: {e}")
            
            # Generate summary
            summary = "Correlation heatmap showing relationships between variables."
            stats = {"columns": df_numeric.columns.tolist()}
            
            # Prepare JSON
            fig_json = fig.to_dict()
            
            return True, "Success", html_path, png_path, chart_id, summary, stats, safe_json_dumps(fig_json)
            
        except Exception as e:
            logger.exception("Heatmap generation failed")
            return False, str(e), "", "", "", "", {}, None
    
    def _create_figure(self, df: pd.DataFrame, x_col: str, y_cols: list, chart_type: str, title: str) -> go.Figure:
        """Create plotly figure based on chart type"""
        
        fig = go.Figure()
        
        x_data = df[x_col].values
        
        for idx, y_col in enumerate(y_cols):
            color = self.COLORS[idx % len(self.COLORS)]
            y_data = df[y_col].values
            
            if chart_type == 'scatter':
                # Drop NaNs for this specific column to avoid empty points
                col_df = df[[x_col, y_col]].dropna()
                fig.add_trace(go.Scatter(
                    x=col_df[x_col].tolist(),
                    y=col_df[y_col].tolist(),
                    mode='markers',
                    name=y_col,
                    marker=dict(
                        color=color,
                        size=5,  # Reduced from 8
                        opacity=0.6,  # Reduced from 0.7
                        line=dict(width=0.5, color='white') # Thinner border
                    ),
                    hovertemplate=f'<b>{y_col}</b><br>{x_col}: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                ))
            
            elif chart_type == 'bar':
                # Smart Aggregation for large datasets
                # If too many points, bar charts look like a solid wall. We aggregate them.
                MAX_BARS = 50
                
                plot_df = df.copy()
                
                if len(plot_df) > MAX_BARS:
                    logger.info(f"Bar chart has {len(plot_df)} points. Aggregating to {MAX_BARS} bins for better visuals.")
                    
                    # Check if X is numeric/datetime or categorical
                    is_numeric_x = pd.api.types.is_numeric_dtype(plot_df[x_col]) or pd.api.types.is_datetime64_any_dtype(plot_df[x_col])
                    
                    if is_numeric_x:
                        # Binning for numeric/time data
                        # Create bins
                        try:
                            # Use pandas cut to bin data
                            # We need to handle datetime explicitly if needed, but often converting to numeric helps
                            if pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
                                plot_df['bin'] = pd.cut(plot_df[x_col], bins=MAX_BARS)
                                # For display, use the center of the bin
                                plot_df[x_col] = plot_df['bin'].apply(lambda x: x.mid)
                            else:
                                plot_df['bin'] = pd.cut(plot_df[x_col], bins=MAX_BARS)
                                plot_df[x_col] = plot_df['bin'].apply(lambda x: x.mid)
                            
                            # Aggregate Y columns (mean is usually best for "trend" in bars, sum for "total")
                            # Defaulting to mean for general analytics, but maybe sum is better? 
                            # Let's use mean for now as it's safer for "average temperature" etc.
                            # If the user wants "total", they usually ask for it.
                            agg_dict = {col: 'mean' for col in y_cols}
                            plot_df = plot_df.groupby(x_col, as_index=False).agg(agg_dict)
                        except Exception as e:
                            logger.warning(f"Binning failed: {e}. Showing top {MAX_BARS} items.")
                            plot_df = plot_df.head(MAX_BARS)
                    else:
                        # Categorical: Show Top N and "Other"
                        # Sort by first Y column to find "top"
                        first_y = y_cols[0]
                        plot_df = plot_df.sort_values(by=first_y, ascending=False)
                        
                        if len(plot_df) > MAX_BARS:
                            top_df = plot_df.iloc[:MAX_BARS-1]
                            other_df = plot_df.iloc[MAX_BARS-1:]
                            
                            # Sum or Mean for "Other"? Usually Sum for categorical (e.g. "Sales by Region")
                            # But for "Temperature by Sensor", Mean is better.
                            # Let's try to guess based on column name? No, too risky.
                            # Let's use Mean as a safe default for scientific data.
                            other_row = {x_col: 'Other (Aggregated)'}
                            for col in y_cols:
                                other_row[col] = other_df[col].mean()
                            
                            plot_df = pd.concat([top_df, pd.DataFrame([other_row])], ignore_index=True)
                
                # Plotting
                for idx, y_col in enumerate(y_cols):
                    # Use a slightly different color for each bar trace if multiple
                    # But usually bar charts with multiple Ys are grouped.
                    # Plotly handles grouping automatically if we just add traces.
                    
                    fig.add_trace(go.Bar(
                        x=plot_df[x_col].tolist(),
                        y=plot_df[y_col].tolist(),
                        name=y_col,
                        marker_color=color if len(y_cols) == 1 else self.COLORS[idx % len(self.COLORS)],
                        marker_line_width=0, # Clean look
                        opacity=0.8,
                        hovertemplate=f'<b>{y_col}</b><br>{x_col}: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                    ))
                
                fig.update_layout(
                    barmode='group', 
                    bargap=0.15,
                    bargroupgap=0.05
                )
            
            elif chart_type == 'histogram':
                fig.add_trace(go.Histogram(
                    x=df[y_col].tolist(),
                    name=y_col,
                    marker_color=color,
                    opacity=0.7,
                    hovertemplate=f'<b>{y_col}</b><br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
                ))
            
            elif chart_type == 'box':
                fig.add_trace(go.Box(
                    y=df[y_col].tolist(),
                    name=y_col,
                    marker_color=color,
                    boxmean='sd',  # Show mean and std deviation
                    hovertemplate=f'<b>{y_col}</b><br>Value: %{{y:.2f}}<extra></extra>'
                ))
            
            elif chart_type == 'area':
                fig.add_trace(go.Scatter(
                    x=df[x_col].tolist(),
                    y=df[y_col].tolist(),
                    mode='lines',
                    name=y_col,
                    line=dict(color=color, width=2),
                    fill='tonexty' if idx > 0 else 'tozeroy',
                    fillcolor=self._rgba_from_hex(color, 0.3),
                    hovertemplate=f'<b>{y_col}</b><br>{x_col}: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                ))

            elif chart_type == 'pie':
                # Pie charts require categorical X-axis, not continuous numeric
                # Check if X-column is suitable for pie chart
                unique_x_values = df[x_col].nunique()
                
                if unique_x_values > 20:
                    # Too many categories - pie chart will be unreadable
                    # Return error with helpful suggestion
                    error_msg = (
                        f"Pie chart unsuitable: X-column '{x_col}' has {unique_x_values} unique values (max 20). "
                        f"Suggestion: Use a Histogram to visualize the distribution of {', '.join(y_cols)}, "
                        f"or use a Line/Scatter chart with '{x_col}' as the X-axis."
                    )
                    logger.warning(error_msg)
                    raise ValueError(error_msg)
                
                # Pie charts with too many slices are unreadable.
                # Aggregate small slices into "Other"
                pie_df = df[[x_col, y_col]].copy()
                pie_df[y_col] = pd.to_numeric(pie_df[y_col], errors='coerce')
                pie_df = pie_df.dropna(subset=[y_col])
                
                # Group by label and sum values
                pie_data = pie_df.groupby(x_col)[y_col].sum().reset_index()
                
                # If too many categories, group small ones
                if len(pie_data) > 15:
                    total = pie_data[y_col].sum()
                    threshold = total * 0.02 # 2% threshold
                    
                    mask = pie_data[y_col] < threshold
                    other_sum = pie_data.loc[mask, y_col].sum()
                    
                    pie_data = pie_data.loc[~mask].copy()
                    if other_sum > 0:
                        new_row = pd.DataFrame([{x_col: 'Other', y_col: other_sum}])
                        pie_data = pd.concat([pie_data, new_row], ignore_index=True)
                
                fig.add_trace(go.Pie(
                    labels=pie_data[x_col].tolist(),
                    values=pie_data[y_col].tolist(),
                    name=y_col,
                    marker=dict(colors=self.COLORS),
                    hovertemplate=f'<b>{y_col}</b><br>{x_col}: %{{label}}<br>Value: %{{value}}<br>Percent: %{{percent}}<extra></extra>'
                ))
            
            else:  # Default line chart
                # Drop NaNs for this specific column to ensure clean line plotting
                col_df = df[[x_col, y_col]].dropna()
                # Force numeric conversion
                col_df[y_col] = pd.to_numeric(col_df[y_col], errors='coerce')
                col_df = col_df.dropna(subset=[y_col])
                
                # Sort by x-axis to ensure proper line drawing
                col_df = col_df.sort_values(by=x_col)
                
                y_vals = col_df[y_col].tolist()
                x_vals = col_df[x_col].tolist()
                
                # For sparse data (many gaps), use markers only
                # For dense data, use lines with optional markers
                data_density = len(col_df) / len(df) if len(df) > 0 else 1.0
                
                if data_density < 0.3:  # Sparse data (less than 30% coverage)
                    # Use markers only to avoid connecting unrelated points
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='markers',
                        name=y_col,
                        marker=dict(size=6, color=color, line=dict(width=1, color='white')),
                        hovertemplate=f'<b>{y_col}</b><br>{x_col}: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                    ))
                else:  # Dense data
                    # Manage markers for large datasets
                    show_markers = len(x_vals) < 100
                    
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers' if show_markers else 'lines',
                        name=y_col,
                        connectgaps=False,  # CRITICAL: Don't connect across gaps
                        line=dict(color=color, width=2.0),
                        marker=dict(size=5, color=color, line=dict(width=1, color='white')) if show_markers else None,
                        hovertemplate=f'<b>{y_col}</b><br>{x_col}: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                    ))
        
        return fig
    
    def _add_range_selector(self, fig: go.Figure):
        """Add interactive range selector for time-series data"""
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=10, label="10pts", step="all", stepmode="backward"),
                    dict(count=50, label="50pts", step="all", stepmode="backward"),
                    dict(count=100, label="100pts", step="all", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="rgba(150, 150, 150, 0.1)",
                activecolor="rgba(0, 119, 187, 0.3)",
                font=dict(color="black")
            )
        )
    
    def _configure_layout(self, fig: go.Figure, title: str):
        """Configure professional layout"""
        # Truncate long titles
        display_title = title
        if len(display_title) > 60:
            display_title = display_title[:57] + "..."

        fig.update_layout(
            title=dict(
                text=f"<b>{display_title}</b>",
                font=dict(size=20, family="Arial, sans-serif", color="#333"),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            xaxis=dict(
                title=dict(font=dict(size=14, family="Arial, sans-serif", color="#333")),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.3)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(0, 0, 0, 0.3)'
            ),
            yaxis=dict(
                title=dict(text="Value", font=dict(size=14, family="Arial, sans-serif", color="#333")),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.3)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(0, 0, 0, 0.3)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                font=dict(size=11)
            ),
            margin=dict(l=80, r=80, t=100, b=120)
        )
    
    def _rgba_from_hex(self, hex_color: str, alpha: float = 1.0) -> str:
        """Convert hex color to rgba string"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'
    
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
                try:
                    slope = np.polyfit(np.arange(len(col_data)), col_data.values, 1)[0]
                    if abs(slope) < 0.01:
                        trend = "stable"
                    elif slope > 0:
                        trend = "increasing"
                    else:
                        trend = "decreasing"
                except Exception:
                    trend = "unknown"
                
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
            parts = [f"Interactive {chart_type} chart showing {', '.join(y_cols)} vs {x_col}."]
            for col, data in stats["columns"].items():
                trend_word = {"increasing": "increases", "decreasing": "decreases", "stable": "remains stable"}.get(data["trend"], "varies")
                parts.append(f"{col} {trend_word}, ranging from {data['min']:.2f} to {data['max']:.2f} (mean: {data['mean']:.2f}).")
            parts.append("Use the range selector to focus on specific data segments.")
            summary = " ".join(parts)
        
        return summary, stats

    def _llm_summary(self, x_col, y_cols, chart_type, stats, user_query):
        """Generate concise AI summary"""
        try:
            prompt = f"""Analyze this {chart_type} chart data:
            X-axis: {x_col}
            Y-axis: {', '.join(y_cols)}
            Statistics: {safe_json_dumps(stats['columns'], indent=2)}
            User Query: {user_query or 'N/A'}
            
            Provide a 2-sentence technical summary highlighting key trends and values.
            """
            response = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Lower temperature for faster, more deterministic output
                max_tokens=60   # Reduced tokens for speed
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM summary failed: {e}")
            return f"Chart showing {', '.join(y_cols)} vs {x_col}."

    def _generate_static_fallback(self, df, x_col, y_cols, chart_type, title, output_path):
        """Generate a static PNG using Matplotlib as a fallback"""
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        
        # Manage markers for large datasets
        show_markers = len(df) < 100
        marker_style = 'o' if show_markers else None
        
        # Truncate title for fallback
        display_title = title
        if len(display_title) > 60:
            display_title = display_title[:57] + "..."

        # Plot based on type
        if chart_type == 'line':
            for col in y_cols:
                sns.lineplot(data=df, x=x_col, y=col, label=col, linewidth=2, marker=marker_style, markersize=4)
        elif chart_type == 'scatter':
            for col in y_cols:
                sns.scatterplot(data=df, x=x_col, y=col, label=col, s=50)
        elif chart_type == 'bar':
            # Melt for seaborn barplot if multiple Ys
            if len(y_cols) > 1:
                df_melt = df.melt(id_vars=[x_col], value_vars=y_cols, var_name='Variable', value_name='Value')
                sns.barplot(data=df_melt, x=x_col, y='Value', hue='Variable')
            else:
                sns.barplot(data=df, x=x_col, y=y_cols[0], color=self.COLORS[0])
        elif chart_type == 'histogram':
            for col in y_cols:
                sns.histplot(data=df, x=col, label=col, kde=True, alpha=0.6)
        elif chart_type == 'box':
            # Melt for boxplot
            df_melt = df.melt(value_vars=y_cols, var_name='Variable', value_name='Value')
            sns.boxplot(data=df_melt, x='Variable', y='Value')
        else:
            # Default to line
            for col in y_cols:
                plt.plot(df[x_col], df[col], label=col, marker=marker_style)
                
        plt.title(display_title, fontsize=16, pad=20)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel("Value", fontsize=12)
        
        if len(y_cols) > 1 or chart_type in ['line', 'scatter', 'histogram']:
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=min(len(y_cols), 4), frameon=True)
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

# Singleton
_plotly_chart_generator = None

def get_plotly_chart_generator(llm_client=None, width: int = 1200, height: int = 700):
    global _plotly_chart_generator
    if _plotly_chart_generator is None:
        _plotly_chart_generator = PlotlyChartGenerator(llm_client, width, height)
    return _plotly_chart_generator
