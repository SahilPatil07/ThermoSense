"""
Comparative Analysis Module for ThermoSense Bot
Compare multiple test runs and generate differential reports
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
import json
import threading
import logging
from backend.tools.json_utils import safe_json_dumps
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

class ComparativeAnalyzer:
    """Compare multiple test datasets"""
    
    def __init__(self):
        self.comparison_methods = ['absolute', 'percentage', 'normalized']
        self.logger = logging.getLogger("comparative_analyzer")
    
    def compare_datasets(self, datasets: Dict[str, pd.DataFrame], 
                        columns: List[str]) -> Dict[str, Any]:
        """
        Compare multiple datasets
        """
        results = {
            'dataset_names': list(datasets.keys()),
            'columns_compared': columns,
            'statistics': {},
            'differences': {},
            'rankings': {},
            'summary': []
        }
        
        for col in columns:
            col_stats = {}
            col_data = {}
            
            # Gather data from all datasets
            for name, df in datasets.items():
                if col in df.columns:
                    data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if not data.empty:
                        col_data[name] = data
                        col_stats[name] = {
                            'mean': float(data.mean()),
                            'median': float(data.median()),
                            'std': float(data.std()),
                            'min': float(data.min()),
                            'max': float(data.max()),
                            'count': int(len(data))
                        }
            
            results['statistics'][col] = col_stats
            
            # Calculate differences
            if len(col_data) >= 2:
                differences = self._calculate_differences(col_data)
                results['differences'][col] = differences
                
                # Rank datasets by mean
                rankings = sorted(col_stats.items(), 
                                key=lambda x: x[1]['mean'], 
                                reverse=True)
                results['rankings'][col] = [r[0] for r in rankings]
        
        # Generate summary insights
        results['summary'] = self._generate_comparison_summary(results)
        
        return results
    
    def _calculate_differences(self, data_dict: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate pairwise differences between datasets"""
        differences = {
            'pairwise': {},
            'max_delta': 0,
            'best_dataset': None,
            'worst_dataset': None
        }
        
        names = list(data_dict.keys())
        means = {name: data.mean() for name, data in data_dict.items()}
        
        # Pairwise comparisons
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                key = f"{name1}_vs_{name2}"
                abs_diff = means[name1] - means[name2]
                pct_diff = (abs_diff / means[name2] * 100) if means[name2] != 0 else 0
                
                differences['pairwise'][key] = {
                    'absolute_difference': float(abs_diff),
                    'percentage_difference': float(pct_diff),
                    'better': name1 if abs_diff > 0 else name2 # "Better" depends on context, assuming higher is better for now
                }
        
        # Overall best/worst (Highest mean = best, for simplicity)
        sorted_means = sorted(means.items(), key=lambda x: x[1])
        if sorted_means:
            differences['worst_dataset'] = sorted_means[0][0]
            differences['best_dataset'] = sorted_means[-1][0]
            differences['max_delta'] = float(sorted_means[-1][1] - sorted_means[0][1])
        
        return differences
    
    def generate_agentic_insight(self, llm_orchestrator, results: Dict[str, Any]) -> str:
        """
        Generate agentic insight using the LLM orchestrator
        """
        if llm_orchestrator:
            return llm_orchestrator.analyze_comparison(results)
        return self._generate_comparison_summary(results)

    def _generate_comparison_summary(self, results: Dict[str, Any]) -> str:
        """Generate text summary of comparison"""
        summary = []
        
        summary.append(f"Comparative analysis of {len(results['dataset_names'])} datasets.")
        
        for col, rankings in results['rankings'].items():
            if rankings:
                best = rankings[0]
                worst = rankings[-1]
                stats = results['statistics'][col]
                
                diff = results['differences'][col]
                max_delta = diff.get('max_delta', 0)
                
                summary.append(f"\n**{col}**:")
                summary.append(f"- Highest: {best} (Mean: {stats[best]['mean']:.2f})")
                summary.append(f"- Lowest: {worst} (Mean: {stats[worst]['mean']:.2f})")
                summary.append(f"- Variation: {max_delta:.2f} between best and worst.")
        
        return "\n".join(summary)
    
    def generate_comparison_chart(self, datasets: Dict[str, pd.DataFrame],
                                 column: str, output_path: str,
                                 chart_type: str = 'line') -> Tuple[bool, str, str, Optional[str]]:
        """
        Generate comparison chart using Plotly
        Returns: (Success, HTML Path, PNG Path, Plotly JSON)
        """
        try:
            fig = go.Figure()
            
            # Prepare output paths
            path_obj = Path(output_path).absolute()
            html_path = str(path_obj.with_suffix('.html'))
            png_path = str(path_obj.with_suffix('.png'))
            
            self.logger.info(f"Generating comparison chart: {html_path}")
            
            if chart_type == 'line':
                for name, df in datasets.items():
                    if column in df.columns:
                        # Try to find time column for X axis
                        x_col = None
                        for t_col in ['Time', 'time', 'Timestamp', 'Date', 'Tme', 'elapsed']:
                            if t_col in df.columns:
                                x_col = t_col
                                break
                        
                        # Prepare data
                        if x_col:
                            df_plot = df[[x_col, column]].dropna().copy()
                            df_plot[column] = pd.to_numeric(df_plot[column], errors='coerce')
                            df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors='coerce')
                            df_plot = df_plot.dropna().sort_values(by=x_col)
                            
                            show_markers = len(df_plot) < 50
                            fig.add_trace(go.Scatter(
                                x=df_plot[x_col].tolist(), 
                                y=df_plot[column].tolist(), 
                                mode='lines+markers' if show_markers else 'lines', 
                                name=name,
                                line=dict(width=2),  # Reduced from 3
                                marker=dict(size=4) if show_markers else None, # Reduced from 6
                                connectgaps=True
                            ))
                        else:
                            y_data = pd.to_numeric(df[column], errors='coerce').dropna().tolist()
                            show_markers = len(y_data) < 50
                            fig.add_trace(go.Scatter(
                                y=y_data, 
                                mode='lines+markers' if show_markers else 'lines', 
                                name=name,
                                line=dict(width=2), # Reduced from 3
                                marker=dict(size=4) if show_markers else None, # Reduced from 6
                                connectgaps=True
                            ))
                            
            elif chart_type == 'bar':
                names = []
                means = []
                stds = []
                
                for name, df in datasets.items():
                    if column in df.columns:
                        data = pd.to_numeric(df[column], errors='coerce').dropna()
                        names.append(name)
                        means.append(data.mean())
                        stds.append(data.std())
                
                fig.add_trace(go.Bar(
                    x=names, y=means,
                    error_y=dict(type='data', array=stds),
                    name=column
                ))
            
            elif chart_type == 'box':
                for name, df in datasets.items():
                    if column in df.columns:
                        data = pd.to_numeric(df[column], errors='coerce').dropna().tolist()
                        fig.add_trace(go.Box(y=data, name=name))
            
            fig.update_layout(
                title=f'Comparison: {column}',
                yaxis_title=column,
                template='plotly_white',
                hovermode='x unified' if chart_type == 'line' else 'closest',
                autosize=True
            )
            
            # Save HTML (interactive)
            fig.write_html(html_path, config={'responsive': True})
            
            # Save PNG (static for reports)
            export_success = False
            try:
                def export_png():
                    nonlocal export_success
                    try:
                        pio.write_image(fig, png_path, format='png', scale=2)
                        export_success = True
                    except Exception as e:
                        self.logger.warning(f"Kaleido comparison export failed: {e}")

                export_thread = threading.Thread(target=export_png)
                export_thread.daemon = True
                export_thread.start()
                export_thread.join(timeout=5.0)
                
                if export_thread.is_alive():
                    self.logger.warning("Kaleido comparison export timed out")
            except Exception as e:
                self.logger.warning(f"PNG generation setup failed: {e}")

            if not export_success or not Path(png_path).exists():
                try:
                    self.logger.info("Generating static comparison fallback using Matplotlib...")
                    self._generate_static_fallback(datasets, column, str(path_obj), chart_type)
                    png_path = str(path_obj.with_suffix('.png'))
                    if Path(png_path).exists():
                        self.logger.info(f"Static fallback success: {png_path}")
                    else:
                        self.logger.error(f"Static fallback failed to create file: {png_path}")
                except Exception as e2:
                    self.logger.error(f"Matplotlib comparison fallback failed: {e2}")
                    png_path = None

            # Prepare responsive JSON
            fig_json = fig.to_dict()
            if 'layout' in fig_json:
                fig_json['layout'].pop('width', None)
                fig_json['layout'].pop('height', None)
                fig_json['layout']['autosize'] = True

            return True, html_path, png_path, safe_json_dumps(fig_json)
        
        except Exception as e:
            self.logger.error(f"Comparison chart error: {e}")
            return False, "", "", None

    def generate_advanced_comparison_chart(self, datasets: Dict[str, pd.DataFrame],
                                         series_config: List[Dict[str, str]],
                                         base_x_column: str,
                                         output_path: str) -> Tuple[bool, str, str, Optional[str]]:
        """
        Generate comparison chart with different Y columns per file
        series_config: List of dicts with 'filename', 'y_column', 'label'
        Returns: (Success, HTML Path, PNG Path, Plotly JSON)
        """
        try:
            fig = go.Figure()
            
            # Prepare output paths
            path_obj = Path(output_path).absolute()
            html_path = str(path_obj.with_suffix('.html'))
            png_path = str(path_obj.with_suffix('.png'))
            
            self.logger.info(f"Generating advanced comparison chart: {html_path}")
            
            for series in series_config:
                fname = series['filename']
                y_col = series['y_column']
                label = series.get('label', f"{fname} - {y_col}")
                
                if fname in datasets:
                    df = datasets[fname]
                    if y_col in df.columns:
                        y_data = pd.to_numeric(df[y_col], errors='coerce').dropna()
                        
                        # X Axis Logic
                        x_data = None
                        # If base_x_column is provided and exists in THIS dataframe, use it
                        if base_x_column and base_x_column in df.columns:
                             x_data = df[base_x_column]
                        elif base_x_column:
                            # Fuzzy match base_x_column
                            from thefuzz import process
                            match = process.extractOne(base_x_column, df.columns.tolist())
                            if match and match[1] >= 90: # High confidence only
                                x_data = df[match[0]]
                            else:
                                # Try to find a time column
                                for t_col in ['Time', 'time', 'Timestamp', 'Date', 'Index']:
                                    if t_col in df.columns:
                                        x_data = df[t_col]
                                        break
                        else:
                            # Try to find a time column
                            for t_col in ['Time', 'time', 'Timestamp', 'Date', 'Index']:
                                if t_col in df.columns:
                                    x_data = df[t_col]
                                    break
                        
                        if x_data is not None and len(x_data) == len(y_data):
                             show_markers = len(x_data) < 50
                             fig.add_trace(go.Scatter(
                                 x=x_data.tolist(), y=y_data.tolist(), 
                                 mode='lines+markers' if show_markers else 'lines', 
                                 name=label,
                                 line=dict(width=2),
                                 marker=dict(size=4) if show_markers else None,
                                 connectgaps=True
                             ))
                        else:
                             show_markers = len(y_data) < 50
                             fig.add_trace(go.Scatter(
                                 y=y_data.tolist(), 
                                 mode='lines+markers' if show_markers else 'lines', 
                                 name=label,
                                 line=dict(width=2),
                                 marker=dict(size=4) if show_markers else None,
                                 connectgaps=True
                             ))
            
            fig.update_layout(
                title="Advanced Comparison", 
                template='plotly_white',
                hovermode='x unified'
            )
            
            # Save HTML (interactive)
            fig.write_html(html_path)
            
            # Save PNG (static for reports)
            export_success = False
            try:
                def export_png_adv():
                    nonlocal export_success
                    try:
                        pio.write_image(fig, png_path, format='png', scale=2)
                        export_success = True
                    except Exception as e:
                        self.logger.warning(f"Kaleido advanced comparison export failed: {e}")

                export_thread = threading.Thread(target=export_png_adv)
                export_thread.daemon = True
                export_thread.start()
                export_thread.join(timeout=5.0)
            except Exception as e:
                self.logger.warning(f"Advanced PNG generation setup failed: {e}")

            if not export_success or not Path(png_path).exists():
                try:
                    self.logger.info("Generating static advanced comparison fallback using Matplotlib...")
                    self._generate_advanced_static_fallback(datasets, series_config, base_x_column, str(path_obj))
                    png_path = str(path_obj.with_suffix('.png'))
                    if Path(png_path).exists():
                        self.logger.info(f"Advanced static fallback success: {png_path}")
                    else:
                        self.logger.error(f"Advanced static fallback failed: {png_path}")
                except Exception as e2:
                    self.logger.error(f"Matplotlib advanced fallback failed: {e2}")
                    png_path = None

            return True, html_path, png_path, safe_json_dumps(fig.to_dict())
        
        except Exception as e:
            self.logger.error(f"Advanced comparison chart error: {e}")
            return False, "", "", None

    def auto_detect_comparable_columns(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Identify columns that exist in all datasets and are numeric
        """
        if not datasets:
            return []
            
        # Get columns from first dataset
        first_df = list(datasets.values())[0]
        common_cols = set(first_df.select_dtypes(include=[np.number]).columns)
        
        # Intersect with others
        for df in datasets.values():
            numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
            common_cols = common_cols.intersection(numeric_cols)
            
        return list(common_cols)

    def _generate_static_fallback(self, datasets, column, output_path, chart_type):
        """Generate static comparison chart using Matplotlib"""
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        
        png_path = str(Path(output_path).with_suffix('.png'))
        
        if chart_type == 'line':
            for name, df in datasets.items():
                if column in df.columns:
                    data = pd.to_numeric(df[column], errors='coerce').dropna()
                    # Try to find time column
                    x_data = None
                    for t_col in ['Time', 'time', 'Timestamp', 'Date']:
                        if t_col in df.columns:
                            x_data = df[t_col]
                            break
                    
                    if x_data is not None and len(x_data) == len(data):
                        plt.plot(x_data, data, label=name)
                    else:
                        plt.plot(data.values, label=name)
                        
        elif chart_type == 'bar':
            names = []
            means = []
            for name, df in datasets.items():
                if column in df.columns:
                    data = pd.to_numeric(df[column], errors='coerce').dropna()
                    names.append(name)
                    means.append(data.mean())
            sns.barplot(x=names, y=means)
            
        elif chart_type == 'box':
            data_list = []
            labels = []
            for name, df in datasets.items():
                if column in df.columns:
                    data_list.append(pd.to_numeric(df[column], errors='coerce').dropna())
                    labels.append(name)
            plt.boxplot(data_list, labels=labels)
            
        plt.title(f'Comparison: {column}', fontsize=16, pad=20)
        plt.ylabel(column, fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=120) # Increased DPI
        plt.close()

    def _generate_advanced_static_fallback(self, datasets, series_config, base_x_column, output_path):
        """Generate static advanced comparison chart using Matplotlib"""
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        
        png_path = str(Path(output_path).with_suffix('.png'))
        
        for series in series_config:
            fname = series['filename']
            y_col = series['y_column']
            label = series.get('label', f"{fname} - {y_col}")
            
            if fname in datasets:
                df = datasets[fname]
                if y_col in df.columns:
                    y_data = pd.to_numeric(df[y_col], errors='coerce').dropna()
                    
                    # Simple X axis detection for fallback
                    x_data = None
                    if base_x_column and base_x_column in df.columns:
                        x_data = df[base_x_column]
                    else:
                        for t_col in ['Time', 'time', 'Timestamp', 'Date', 'Index']:
                            if t_col in df.columns:
                                x_data = df[t_col]
                                break
                    
                    if x_data is not None and len(x_data) == len(y_data):
                        plt.plot(x_data, y_data, label=label, linewidth=2, marker='o', markersize=3)
                    else:
                        plt.plot(y_data.values, label=label, linewidth=2, marker='o', markersize=3)
        
        plt.title("Advanced Comparison", fontsize=16, pad=20)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close()
