"""
Advanced Analytics Engine for Thermal Data Analysis
Provides deep insights, anomaly detection, root cause analysis, and pattern recognition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)


class ThermalAnalytics:
    """
    Advanced analytics for thermal test data
    Provides insights beyond basic statistics
    """
    
    def __init__(self, llm_client: Any = None):
        self.llm = llm_client
        self.anomaly_threshold = 3  # Standard deviations for outlier detection
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a dataframe and identify critical columns
        Returns analysis with 'critical_columns' key
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {'critical_columns': numeric_cols}
        
        # Find time column for X-axis
        x_col = None
        for time_col in ['Time', 'Tm_Sampling', 'Timestamp', 'time', 'timestamp']:
            if time_col in df.columns:
                x_col = time_col
                break
        
        if not x_col:
            x_col = numeric_cols[0]
        
        # Get Y columns (exclude X column)
        y_cols = [col for col in numeric_cols if col != x_col]
        
        if not y_cols:
            return {'critical_columns': numeric_cols[:3]}
        
        # Calculate variance for each column to find critical ones
        variances = {}
        for col in y_cols:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 0:
                variances[col] = data.std()
        
        # Sort by variance (highest variance = most critical)
        sorted_cols = sorted(variances.items(), key=lambda x: x[1], reverse=True)
        critical_cols = [col for col, _ in sorted_cols[:3]]  # Top 3
        
        # Run comprehensive analysis
        analysis = self.comprehensive_analysis(df, x_col, critical_cols)
        analysis['critical_columns'] = critical_cols
        
        return analysis
    
    def comprehensive_analysis(self, df: pd.DataFrame, x_col: str, y_cols: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on thermal data
        Returns deep insights, not just surface statistics
        """
        analysis = {
            'basic_stats': {},
            'anomalies': {},
            'trends': {},
            'correlations': {},
            'critical_points': {},
            'patterns': {},
            'recommendations': []
        }
        
        try:
            # Basic statistics
            for col in y_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    data = df[col].dropna()
                    analysis['basic_stats'][col] = {
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'range': float(data.max() - data.min()),
                        'q25': float(data.quantile(0.25)),
                        'q75': float(data.quantile(0.75)),
                        'q75': float(data.quantile(0.75)),
                        'coefficient_of_variation': float(data.std() / data.mean()) if data.mean() != 0 else 0.0
                    }
            
            # Anomaly Detection
            analysis['anomalies'] = self._detect_anomalies(df, y_cols)
            
            # Trend Analysis
            analysis['trends'] = self._analyze_trends(df, x_col, y_cols)
            
            # Critical Points
            analysis['critical_points'] = self._find_critical_points(df, x_col, y_cols)
            
            # Pattern Recognition
            analysis['patterns'] = self._detect_patterns(df, x_col, y_cols)
            
            # Correlations between Y columns
            if len(y_cols) > 1:
                analysis['correlations'] = self._analyze_correlations(df, y_cols)
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            
        return analysis
    
    def _detect_anomalies(self, df: pd.DataFrame, y_cols: List[str]) -> Dict[str, Any]:
        """
        Detect anomalies using multiple methods:
        - Z-score method
        - IQR method
        - Rate of change spikes
        """
        anomalies = {}
        
        for col in y_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            data = df[col].dropna()
            col_anomalies = {
                'z_score_outliers': [],
                'iqr_outliers': [],
                'sudden_changes': [],
                'count': 0
            }
            
            # Z-score method
            if len(data) > 3:
                z_scores = np.abs(stats.zscore(data))
                outlier_indices = np.where(z_scores > self.anomaly_threshold)[0]
                if len(outlier_indices) > 0:
                    col_anomalies['z_score_outliers'] = [
                        {
                            'index': int(idx),
                            'value': float(data.iloc[idx]),
                            'z_score': float(z_scores[idx])
                        }
                        for idx in outlier_indices[:10]  # Limit to top 10
                    ]
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            if len(iqr_outliers) > 0:
                col_anomalies['iqr_outliers'] = [
                    {'index': int(idx), 'value': float(val)}
                    for idx, val in list(iqr_outliers.items())[:10]
                ]
            
            # Sudden changes (rate of change)
            if len(data) > 1:
                rate_of_change = data.diff().abs()
                change_threshold = rate_of_change.mean() + 3 * rate_of_change.std()
                sudden_changes = rate_of_change[rate_of_change > change_threshold]
                if len(sudden_changes) > 0:
                    col_anomalies['sudden_changes'] = [
                        {
                            'index': int(idx),
                            'change': float(val),
                            'from_value': float(data.iloc[idx-1]) if idx > 0 else None,
                            'to_value': float(data.iloc[idx])
                        }
                        for idx, val in list(sudden_changes.items())[:10]
                    ]
            
            col_anomalies['count'] = (
                len(col_anomalies['z_score_outliers']) +
                len(col_anomalies['iqr_outliers']) +
                len(col_anomalies['sudden_changes'])
            )
            
            anomalies[col] = col_anomalies
        
        return anomalies
    
    def _analyze_trends(self, df: pd.DataFrame, x_col: str, y_cols: List[str]) -> Dict[str, Any]:
        """
        Analyze trends using statistical methods
        """
        trends = {}
        
        for col in y_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            data = df[col].dropna()
            if len(data) < 3:
                continue
            
            # Linear regression
            x_values = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, data.values)
            
            # Determine trend direction and strength
            if abs(slope) < 0.001:
                direction = "stable"
                strength = "none"
            elif slope > 0:
                direction = "increasing"
                strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.4 else "weak"
            else:
                direction = "decreasing"
                strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.4 else "weak"
            
            # Detect non-linearity
            # Quadratic fit
            if len(data) > 5:
                try:
                    coeffs = np.polyfit(x_values, data.values, 2)
                    poly_values = np.polyval(coeffs, x_values)
                    
                    # Calculate R-squared for polynomial fit
                    ss_res = np.sum((data.values - poly_values)**2)
                    ss_tot = np.sum((data.values - data.mean())**2)
                    
                    if ss_tot > 1e-10: # Avoid division by zero
                        poly_r2 = 1 - (ss_res / ss_tot)
                        is_nonlinear = poly_r2 > (r_value**2 + 0.1)
                    else:
                        is_nonlinear = False
                except Exception:
                    is_nonlinear = False
            else:
                is_nonlinear = False
            
            trends[col] = {
                'direction': direction,
                'strength': strength,
                'slope': float(slope),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'is_significant': bool(p_value < 0.05),
                'is_nonlinear': bool(is_nonlinear),
                'rate_of_change_per_unit': float(slope)
            }
        
        return trends
    
    def _find_critical_points(self, df: pd.DataFrame, x_col: str, y_cols: List[str]) -> Dict[str, Any]:
        """
        Find peaks, valleys, inflection points
        """
        critical_points = {}
        
        for col in y_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            data = df[col].dropna().values
            if len(data) < 5:
                continue
            
            col_critical = {
                'peaks': [],
                'valleys': [],
                'max_point': None,
                'min_point': None
            }
            
            # Find peaks
            peaks, properties = find_peaks(data, prominence=data.std()/2)
            if len(peaks) > 0:
                col_critical['peaks'] = [
                    {
                        'index': int(idx),
                        'value': float(data[idx]),
                        'prominence': float(properties['prominences'][i]) if 'prominences' in properties else None
                    }
                    for i, idx in enumerate(peaks[:5])  # Top 5 peaks
                ]
            
            # Find valleys (peaks of inverted signal)
            valleys, v_properties = find_peaks(-data, prominence=data.std()/2)
            if len(valleys) > 0:
                col_critical['valleys'] = [
                    {
                        'index': int(idx),
                        'value': float(data[idx]),
                        'prominence': float(v_properties['prominences'][i]) if 'prominences' in v_properties else None
                    }
                    for i, idx in enumerate(valleys[:5])  # Top 5 valleys
                ]
            
            # Max and min points
            max_idx = np.argmax(data)
            min_idx = np.argmin(data)
            
            col_critical['max_point'] = {
                'index': int(max_idx),
                'value': float(data[max_idx])
            }
            col_critical['min_point'] = {
                'index': int(min_idx),
                'value': float(data[min_idx])
            }
            
            critical_points[col] = col_critical
        
        return critical_points
    
    def _detect_patterns(self, df: pd.DataFrame, x_col: str, y_cols: List[str]) -> Dict[str, Any]:
        """
        Detect common patterns in thermal data:
        - Steady state
        - Exponential rise/decay
        - Oscillations
        - Step changes
        """
        patterns = {}
        
        for col in y_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            data = df[col].dropna().values
            if len(data) < 10:
                continue
            
            col_patterns = []
            
            # Check for steady state (low variance in segments)
            segment_size = max(len(data) // 10, 5)
            segments = [data[i:i+segment_size] for i in range(0, len(data), segment_size) if len(data[i:i+segment_size]) >= 3]
            steady_segments = [i for i, seg in enumerate(segments) if np.std(seg) < data.std() * 0.2]
            
            if len(steady_segments) > 0:
                col_patterns.append({
                    'type': 'steady_state',
                    'description': f'Steady state detected in {len(steady_segments)} segments',
                    'confidence': 'high' if len(steady_segments) > len(segments) * 0.5 else 'medium'
                })
            
            # Check for exponential pattern
            if len(data) > 20:
                try:
                    # Try exponential fit
                    x_vals = np.arange(len(data))
                    # Avoid log of zero/negative
                    if np.all(data > 0):
                        log_data = np.log(data)
                        slope, _, r_val, _, _ = stats.linregress(x_vals, log_data)
                        if r_val**2 > 0.8:
                            pattern_type = 'exponential_growth' if slope > 0 else 'exponential_decay'
                            col_patterns.append({
                                'type': pattern_type,
                                'description': f'{pattern_type.replace("_", " ").title()} pattern',
                                'confidence': 'high' if r_val**2 > 0.9 else 'medium',
                                'rate': float(slope)
                            })
                except:
                    pass
            
            # Check for oscillations (using autocorrelation)
            if len(data) > 50 and data.std() > 1e-10: # Ensure non-constant data
                try:
                    # Normalize data first
                    norm_data = data - data.mean()
                    autocorr = np.correlate(norm_data, norm_data, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    if autocorr[0] != 0:
                        autocorr = autocorr / autocorr[0]  # Normalize
                        
                        # Find peaks in autocorrelation
                        ac_peaks, _ = find_peaks(autocorr[1:], height=0.5)
                        if len(ac_peaks) > 1:
                            # Estimate period
                            period = np.diff(ac_peaks).mean()
                            col_patterns.append({
                                'type': 'oscillation',
                                'description': 'Periodic oscillation detected',
                                'confidence': 'high' if len(ac_peaks) > 2 else 'medium',
                                'estimated_period': float(period)
                            })
                except Exception as e:
                    logger.warning(f"Oscillation detection failed for {col}: {e}")
            
            # Check for step changes
            diff_data = np.abs(np.diff(data))
            threshold = diff_data.mean() + 2 * diff_data.std()
            step_indices = np.where(diff_data > threshold)[0]
            
            if len(step_indices) > 0:
                col_patterns.append({
                    'type': 'step_changes',
                    'description': f'{len(step_indices)} step change(s) detected',
                    'confidence': 'high',
                    'locations': [int(idx) for idx in step_indices[:5]]
                })
            
            patterns[col] = col_patterns
        
        return patterns
    
    def _analyze_correlations(self, df: pd.DataFrame, y_cols: List[str]) -> Dict[str, Any]:
        """
        Analyze correlations between Y columns
        """
        correlations = {}
        
        numeric_df = df[y_cols].select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return correlations
        
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.6:
                    strong_corr.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': float(corr_val),
                        'type': 'positive' if corr_val > 0 else 'negative',
                        'strength': 'very_strong' if abs(corr_val) > 0.8 else 'strong'
                    })
        
        correlations['strong_correlations'] = sorted(strong_corr, key=lambda x: abs(x['correlation']), reverse=True)
        
        return correlations
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on analysis
        """
        recommendations = []
        
        # Check for anomalies
        for col, anomaly_data in analysis['anomalies'].items():
            if anomaly_data['count'] > 0:
                recommendations.append(
                    f"⚠️ {col}: {anomaly_data['count']} anomalies detected. Review data quality and sensor calibration."
                )
        
        # Check for strong trends
        for col, trend_data in analysis['trends'].items():
            if trend_data['is_significant'] and trend_data['strength'] in ['strong', 'moderate']:
                if trend_data['direction'] == 'increasing':
                    recommendations.append(
                        f"📈 {col}: Strong increasing trend detected. Monitor for thermal runaway or degradation."
                    )
                elif trend_data['direction'] == 'decreasing':
                    recommendations.append(
                        f"📉 {col}: Strong decreasing trend detected. Verify cooling system effectiveness."
                    )
        
        # Check for correlations
        if 'strong_correlations' in analysis.get('correlations', {}):
            for corr in analysis['correlations']['strong_correlations'][:3]:
                recommendations.append(
                    f"🔗 {corr['variable1']} and {corr['variable2']}: {corr['strength'].replace('_', ' ').title()} {corr['type']} correlation ({corr['correlation']:.2f})"
                )
        
        return recommendations
    
    def query_data_context(self, df: pd.DataFrame, x_col: str, y_cols: List[str], query: str) -> str:
        """
        Answer specific questions about the data using LLM to generate Pandas queries
        Example: "where is temperature less when speed is above 60?"
        """
        query_lower = query.lower()
        
        # 1. Try LLM-based query generation if available
        if self.llm:
            try:
                cols_info = ", ".join(df.columns.tolist())
                prompt = f"""You are a data analysis expert. Translate this natural language query into a valid Pandas query string.

**Query:** "{query}"
**Available Columns:** {cols_info}

**Guidelines:**
1. Return ONLY the query string that can be used in `df.query()`.
2. Use backticks for column names with spaces or special characters (e.g., `Coolant Temp`).
3. Handle comparative conditions (above, below, greater, less).
4. If the query is not about filtering data, return "NONE".

**Example Output:**
`Coolant Temp` > 80 and `Vehicle Speed` < 20

Pandas Query:"""
                
                resp = self.llm.chat.completions.create(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100
                )
                pandas_query = resp.choices[0].message.content.strip().replace("Pandas Query:", "").strip()
                
                if pandas_query and pandas_query != "NONE":
                    logger.info(f"Generated Pandas query: {pandas_query}")
                    filtered_df = df.query(pandas_query)
                    
                    if not filtered_df.empty:
                        result_parts = [f"Found {len(filtered_df)} data points where: {pandas_query}"]
                        
                        # Add summary of filtered data
                        for col in y_cols:
                            if col in filtered_df.columns:
                                col_data = filtered_df[col].dropna()
                                if not col_data.empty:
                                    result_parts.append(
                                        f"  - {col}: min={col_data.min():.2f}, max={col_data.max():.2f}, mean={col_data.mean():.2f}"
                                    )
                        
                        if x_col in filtered_df.columns:
                            x_values = filtered_df[x_col].dropna()
                            if not x_values.empty:
                                result_parts.append(f"  - {x_col} range: {x_values.min():.2f} to {x_values.max():.2f}")
                        
                        return "\n".join(result_parts)
                    else:
                        return f"No data points match the criteria: {pandas_query}"
            except Exception as e:
                logger.warning(f"LLM query generation failed: {e}. Falling back to keyword parsing.")

        # 2. Fallback to keyword-based parsing (existing logic)
        try:
            result_parts = []
            conditions = []
            # ... (rest of the existing logic)
            # I'll keep the existing logic as a robust fallback
            for col in y_cols + [x_col]:
                if col.lower() in query_lower:
                    # Check for comparative conditions
                    if any(word in query_lower for word in ['above', 'greater', 'more than', '>']):
                        words = query_lower.split()
                        for i, word in enumerate(words):
                            if word in ['above', 'greater'] and i + 1 < len(words):
                                try:
                                    threshold = float(words[i + 1])
                                    filtered = df[df[col] > threshold]
                                    conditions.append((col, '>', threshold, filtered))
                                except: pass
                    
                    if any(word in query_lower for word in ['below', 'less', 'lower than', '<']):
                        words = query_lower.split()
                        for i, word in enumerate(words):
                            if word in ['below', 'less', 'lower'] and i + 1 < len(words):
                                try:
                                    threshold = float(words[i + 1])
                                    filtered = df[df[col] < threshold]
                                    conditions.append((col, '<', threshold, filtered))
                                except: pass
            
            if conditions:
                filtered_df = df.copy()
                for col, op, threshold, _ in conditions:
                    if op == '>':
                        filtered_df = filtered_df[filtered_df[col] > threshold]
                    else:
                        filtered_df = filtered_df[filtered_df[col] < threshold]
                
                if not filtered_df.empty:
                    result_parts.append(f"Found {len(filtered_df)} data points matching your criteria:")
                    for col in y_cols:
                        if col in filtered_df.columns:
                            col_data = filtered_df[col].dropna()
                            if not col_data.empty:
                                result_parts.append(f"  - {col}: min={col_data.min():.2f}, max={col_data.max():.2f}, mean={col_data.mean():.2f}")
                    if x_col in filtered_df.columns:
                        x_values = filtered_df[x_col].dropna()
                        if not x_values.empty:
                            result_parts.append(f"  - {x_col} range: {x_values.min():.2f} to {x_values.max():.2f}")
                    return "\n".join(result_parts)
                else:
                    return "No data points match your criteria."
            
            return "I couldn't parse that query. Please rephrase (e.g., 'where is temperature above 80?')."
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Error processing query: {str(e)}"
