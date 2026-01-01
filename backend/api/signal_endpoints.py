"""
Signal recommendation endpoint
Analyzes uploaded files to recommend critical parameters for visualization
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/signals", tags=["signals"])


class SignalRecommendationRequest(BaseModel):
    """Request for signal recommendations"""
    session_id: str = "default"
    filename: str = None  # Optional: analyze specific file, otherwise all files
    top_n: int = 5  # Number of recommendations to return


class SignalRecommendation(BaseModel):
    """Single signal recommendation"""
    name: str
    score: float  # 0-1 score
    reason: str
    file_source: str = None
    statistics: Dict[str, Any] = {}


class SignalRecommendationResponse(BaseModel):
    """Response with signal recommendations"""
    success: bool
    recommendations: List[SignalRecommendation]
    total_signals: int
    error: str = None


def analyze_signal_importance(df: pd.DataFrame, column: str, file_source: str = None) -> Dict[str, Any]:
    """
    Analyze a signal/parameter to determine its importance for visualization
    
    Scoring criteria:
    - Variance/deviation (high variance = more interesting)
    - Data completeness (fewer nulls = better)
    - Keywords (temp, pressure, power, etc.)
    - Value range diversity
    """
    try:
        series = df[column]
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(series):
            return None
        
        # Calculate statistics
        completeness = 1.0 - (series.isna().sum() / len(series))
        if completeness < 0.5:  # Skip columns with >50% nulls
            return None
        
        # Variance score (normalized)
        values = series.dropna()
        if len(values) < 2:
            return None
            
        std_dev = values.std()
        mean_val = values.mean()
        
        # Coefficient of variation (normalize variance by mean)
        if abs(mean_val) > 1e-10:
            cv = abs(std_dev / mean_val)
        else:
            cv = std_dev  # If mean is near zero, use std dev directly
        
        # Normalize CV to 0-1 scale (cap at high values)
        variance_score = min(cv / 10.0, 1.0)
        
        # Keyword importance
        keywords = {
            'temperature': 1.0, 'temp': 1.0, 'thermal': 1.0,
            'pressure': 0.9, 'power': 0.9, 'voltage': 0.9, 'current': 0.9,
            'speed': 0.8, 'rpm': 0.8, 'velocity': 0.8,
            'flow': 0.7, 'rate': 0.7,
            'critical': 1.0, 'important': 0.9, 'key': 0.8
        }
        
        keyword_score = 0.0
        col_lower = column.lower()
        for keyword, weight in keywords.items():
            if keyword in col_lower:
                keyword_score = max(keyword_score, weight)
        
        # Value range diversity (number of unique values relative to total)
        unique_ratio = len(values.unique()) / len(values)
        diversity_score = min(unique_ratio * 2.0, 1.0)  # Cap at 1.0
        
        # Combined score (weighted average)
        final_score = (
            variance_score * 0.4 +      # Variance is most important
            completeness * 0.2 +         # Data quality matters
            keyword_score * 0.3 +        # Domain relevance
            diversity_score * 0.1        # Value diversity
        )
        
        # Generate reason
        reasons = []
        if variance_score > 0.5:
            reasons.append(f"high variability (CV: {cv:.2f})")
        if keyword_score > 0:
            reasons.append("critical parameter type")
        if completeness == 1.0:
            reasons.append("complete data")
        elif completeness > 0.9:
            reasons.append("mostly complete")
        
        reason = ", ".join(reasons) if reasons else "statistical analysis"
        
        return {
            "name": column,
            "score": final_score,
            "reason": reason,
            "file_source": file_source,
            "statistics": {
                "mean": float(mean_val),
                "std": float(std_dev),
                "min": float(values.min()),
                "max": float(values.max()),
                "completeness": float(completeness),
                "variance_score": float(variance_score),
                "keyword_score": float(keyword_score)
            }
        }
    except Exception as e:
        logger.warning(f"Error analyzing signal {column}: {e}")
        return None


@router.post("/recommend", response_model=SignalRecommendationResponse)
async def recommend_signals(req: SignalRecommendationRequest):
    """
    Recommend critical signals/parameters for visualization
    """
    try:
        from backend.tools.storage import SessionStorage
        from backend.tools.sensor_harvester import SensorHarvester
        
        storage = SessionStorage()
        harvester = SensorHarvester()
        
        # Get files to analyze
        if req.filename:
            files_to_analyze = [req.filename]
        else:
            files_to_analyze = storage.get_uploads(req.session_id)
        
        if not files_to_analyze:
            return SignalRecommendationResponse(
                success=False,
                recommendations=[],
                total_signals=0,
                error="No files uploaded"
            )
        
        # Analyze all signals from all files
        all_recommendations = []
        
        for filename in files_to_analyze:
            file_path = storage.find_file(filename)
            if not file_path or not file_path.exists():
                continue
            
            try:
                # Load file
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif filename.endswith(('.xlsx', '.xls')):
                    from backend.tools.excel_tools import list_sheetnames, read_excel_table
                    sheets = list_sheetnames(str(file_path))
                    if sheets:
                        df = read_excel_table(str(file_path), sheets[0])
                    else:
                        continue
                else:
                    continue
                
                # Get numeric columns only
                _, time_column, numeric_columns = harvester.get_file_columns(str(file_path))
                
                # Analyze each numeric column
                for col in numeric_columns:
                    if col == time_column:  # Skip time column
                        continue
                    
                    analysis = analyze_signal_importance(df, col, file_source=filename)
                    if analysis:
                        all_recommendations.append(analysis)
                
            except Exception as e:
                logger.warning(f"Error analyzing file {filename}: {e}")
                continue
        
        if not all_recommendations:
            return SignalRecommendationResponse(
                success=False,
                recommendations=[],
                total_signals=0,
                error="No numeric signals found in uploaded files"
            )
        
        # Sort by score and take top N
        all_recommendations.sort(key=lambda x: x['score'], reverse=True)
        top_recommendations = all_recommendations[:req.top_n]
        
        # Convert to response model
        recommendations = [
            SignalRecommendation(**rec) for rec in top_recommendations
        ]
        
        return SignalRecommendationResponse(
            success=True,
            recommendations=recommendations,
            total_signals=len(all_recommendations)
        )
        
    except Exception as e:
        logger.exception("Signal recommendation failed")
        return SignalRecommendationResponse(
            success=False,
            recommendations=[],
            total_signals=0,
            error=str(e)
        )


@router.post("/select")
async def save_signal_selection(session_id: str, signals: List[str]):
    """Save user's selected signals for the session"""
    try:
        from backend.tools.storage import SessionStorage
        storage = SessionStorage()
        
        # Store selected signals in session metadata
        session_dir = storage.get_session_dir(session_id)
        selection_file = session_dir / "selected_signals.json"
        
        import json
        with open(selection_file, 'w') as f:
            json.dump({"signals": signals, "timestamp": pd.Timestamp.now().isoformat()}, f)
        
        return JSONResponse({
            "success": True,
            "message": f"Saved {len(signals)} selected signals"
        })
    except Exception as e:
        logger.error(f"Failed to save signal selection: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/selected")
async def get_selected_signals(session_id: str):
    """Get user's selected signals for the session"""
    try:
        from backend.tools.storage import SessionStorage
        storage = SessionStorage()
        
        session_dir = storage.get_session_dir(session_id)
        selection_file = session_dir / "selected_signals.json"
        
        if not selection_file.exists():
            return JSONResponse({
                "success": True,
                "signals": [],
                "message": "No signals selected yet"
            })
        
        import json
        with open(selection_file, 'r') as f:
            data = json.load(f)
        
        return JSONResponse({
            "success": True,
            "signals": data.get("signals", []),
            "timestamp": data.get("timestamp")
        })
    except Exception as e:
        logger.error(f"Failed to get selected signals: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
