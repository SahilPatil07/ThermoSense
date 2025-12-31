"""
SQLAlchemy ORM Models for ThermoSense
Replaces JSON-based storage with relational database
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, 
    ForeignKey, Boolean, Float, JSON, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from backend.db.database import Base


class ChatSession(Base):
    """Chat session model"""
    __tablename__ = "sessions"
    
    id = Column(String(36), primary_key=True)  # UUID
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    uploads = relationship("Upload", back_populates="chat_session", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="chat_session", cascade="all, delete-orphan")
    charts = relationship("Chart", back_populates="chat_session", cascade="all, delete-orphan")
    report_sections = relationship("ReportSection", back_populates="chat_session", cascade="all, delete-orphan")
    tool_executions = relationship("ToolExecution", back_populates="chat_session", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="chat_session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_created_at', 'created_at'),
        Index('idx_session_updated_at', 'updated_at'),
    )
    
    def __repr__(self):
        return f"<ChatSession(id={self.id}, title={self.title})>"


class Upload(Base):
    """Uploaded file tracking"""
    __tablename__ = "uploads"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)  # Relative path in workspace
    file_hash = Column(String(64), nullable=True)  # SHA256 hash for deduplication
    file_size = Column(Integer, nullable=True)  # Size in bytes
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    chat_session = relationship("ChatSession", back_populates="uploads")
    
    # Indexes
    __table_args__ = (
        Index('idx_upload_session_id', 'session_id'),
        Index('idx_upload_filename', 'filename'),
        Index('idx_upload_file_hash', 'file_hash'),
    )
    
    def __repr__(self):
        return f"<Upload(id={self.id}, filename={self.filename})>"


class Message(Base):
    """Chat message history"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    chat_session = relationship("ChatSession", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index('idx_message_session_id', 'session_id'),
        Index('idx_message_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role})>"


class Chart(Base):
    """Generated chart tracking"""
    __tablename__ = "charts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    chart_id = Column(String(36), unique=True, nullable=False)  # UUID for external reference
    chart_type = Column(String(50), nullable=False)  # 'line', 'scatter', 'heatmap', etc.
    
    # Chart configuration (stored as JSON/JSONB)
    config = Column(JSON, nullable=True)  # x_column, y_columns, filename, etc.
    
    # File paths
    html_path = Column(String(512), nullable=True)
    png_path = Column(String(512), nullable=True)
    plotly_json_path = Column(String(512), nullable=True)
    
    # Feedback
    feedback = Column(String(20), nullable=True)  # 'approved', 'rejected', None
    feedback_timestamp = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    generation_time = Column(Float, nullable=True)  # Time in seconds
    
    # Relationships
    chat_session = relationship("ChatSession", back_populates="charts")
    
    # Indexes
    __table_args__ = (
        Index('idx_chart_session_id', 'session_id'),
        Index('idx_chart_chart_id', 'chart_id'),
        Index('idx_chart_feedback', 'feedback'),
        Index('idx_chart_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Chart(id={self.id}, chart_id={self.chart_id}, type={self.chart_type})>"


class ReportSection(Base):
    """Report content organized by section"""
    __tablename__ = "report_sections"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    section_name = Column(String(100), nullable=False)  # 'executive_summary', 'test_results', etc.
    
    # Items stored as JSON array: [{"type": "text", "content": "..."}, {"type": "chart", "chart_id": "..."}]
    items = Column(JSON, nullable=False, default=list)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    chat_session = relationship("ChatSession", back_populates="report_sections")
    
    # Indexes
    __table_args__ = (
        Index('idx_report_section_session_id', 'session_id'),
        Index('idx_report_section_name', 'section_name'),
    )
    
    def __repr__(self):
        return f"<ReportSection(id={self.id}, section={self.section_name})>"


class ToolExecution(Base):
    """Tool execution logs for observability"""
    __tablename__ = "tool_executions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    tool_name = Column(String(100), nullable=False)
    
    # Parameters and results (stored as JSON/JSONB)
    params = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    
    # Execution metadata
    status = Column(String(20), nullable=False)  # 'success', 'failure', 'pending'
    error_message = Column(Text, nullable=True)
    duration = Column(Float, nullable=True)  # Time in seconds
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    chat_session = relationship("ChatSession", back_populates="tool_executions")
    
    # Indexes
    __table_args__ = (
        Index('idx_tool_exec_session_id', 'session_id'),
        Index('idx_tool_exec_tool_name', 'tool_name'),
        Index('idx_tool_exec_status', 'status'),
        Index('idx_tool_exec_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<ToolExecution(id={self.id}, tool={self.tool_name}, status={self.status})>"


class Feedback(Base):
    """User feedback tracking"""
    __tablename__ = "feedbacks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    chart_id = Column(String(36), nullable=True)  # Reference to chart if feedback is chart-specific
    feedback_type = Column(String(20), nullable=False)  # 'thumbs_up', 'thumbs_down'
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    chat_session = relationship("ChatSession", back_populates="feedbacks")
    
    # Indexes
    __table_args__ = (
        Index('idx_feedback_session_id', 'session_id'),
        Index('idx_feedback_chart_id', 'chart_id'),
        Index('idx_feedback_type', 'feedback_type'),
        Index('idx_feedback_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, type={self.feedback_type})>"


# For future pgvector integration
class KnowledgeChunk(Base):
    """Knowledge base chunks with vector embeddings"""
    __tablename__ = "knowledge_chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_file = Column(String(512), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Vector embedding (requires pgvector extension)
    # embedding = Column(Vector(768))  # Uncomment when pgvector is available
    
    # Metadata
    chunk_metadata = Column(JSON, nullable=True)  # File type, page number, etc.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_knowledge_source_file', 'source_file'),
        # Index('idx_knowledge_embedding', 'embedding', postgresql_using='ivfflat'),  # pgvector index
    )
    
    def __repr__(self):
        return f"<KnowledgeChunk(id={self.id}, source={self.source_file})>"
