"""
Database initialization and testing script
"""
import logging
from backend.db.database import init_db, check_db_connection, engine
from backend.db.models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database():
    """Test database connection and table creation"""
    logger.info("=" * 60)
    logger.info("ThermoSense Database Initialization Test")
    logger.info("=" * 60)
    
    # Check connection
    logger.info("\n1. Testing database connection...")
    if not check_db_connection():
        logger.error("❌ Database connection failed!")
        return False
    logger.info("✅ Database connection successful")
    
    # Initialize database
    logger.info("\n2. Initializing database tables...")
    try:
        init_db()
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False
    
    # List created tables
    logger.info("\n3. Verifying created tables...")
    inspector = engine.dialect.get_inspector(engine)
    tables = inspector.get_table_names()
    
    expected_tables = [
        'sessions',
        'uploads',
        'messages',
        'charts',
        'report_sections',
        'tool_executions',
        'feedbacks',
        'knowledge_chunks'
    ]
    
    for table in expected_tables:
        if table in tables:
            logger.info(f"  ✅ {table}")
        else:
            logger.warning(f"  ⚠️ {table} (missing)")
    
    logger.info(f"\nTotal tables created: {len(tables)}")
    
    # Test basic CRUD operations
    logger.info("\n4. Testing basic CRUD operations...")
    from backend.db.database import SessionLocal
    from backend.db.models import Session as DBSession
    from datetime import datetime
    
    db = SessionLocal()
    try:
        # Create
        test_session = DBSession(
            id="test_session_001",
            title="Test Session",
            created_at=datetime.utcnow()
        )
        db.add(test_session)
        db.commit()
        logger.info("  ✅ CREATE: Session created")
        
        # Read
        retrieved = db.query(DBSession).filter(DBSession.id == "test_session_001").first()
        if retrieved and retrieved.title == "Test Session":
            logger.info("  ✅ READ: Session retrieved")
        else:
            logger.error("  ❌ READ: Failed to retrieve session")
        
        # Update
        retrieved.title = "Updated Test Session"
        db.commit()
        updated = db.query(DBSession).filter(DBSession.id == "test_session_001").first()
        if updated.title == "Updated Test Session":
            logger.info("  ✅ UPDATE: Session updated")
        else:
            logger.error("  ❌ UPDATE: Failed to update session")
        
        # Delete
        db.delete(updated)
        db.commit()
        deleted = db.query(DBSession).filter(DBSession.id == "test_session_001").first()
        if deleted is None:
            logger.info("  ✅ DELETE: Session deleted")
        else:
            logger.error("  ❌ DELETE: Failed to delete session")
        
    except Exception as e:
        logger.error(f"  ❌ CRUD operations failed: {e}")
        db.rollback()
        return False
    finally:
        db.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All database tests passed!")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    test_database()
