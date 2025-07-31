"""
SQLite3 Database Manager for MCP Memory and Task Management
Handles memories, knowledge graphs, tasks, and project data
"""

import sqlite3
from sqlite3 import Cursor
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import logging
from functools import wraps

def retry_on_failure(max_retries=3, delay=1.0):
    """Decorator to retry database operations on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

class DatabaseManager:
    def __init__(self, db_path: str = "mcp_memory.db"):
        """Initialize database connection and create tables if they don't exist"""
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self.setup_database()
        
    def _check_connection(self) -> bool:
        """Check if database connection is valid"""
        if not self.connection:
            logging.error("Database connection not established")
            return False
        return True
        
    def setup_database(self) -> None:
        """Create database connection and initialize tables"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            self.create_tables()
            logging.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logging.error(f"Failed to initialize database: {e}")
            raise
            
    def create_tables(self) -> None:
        """Create all necessary tables for the MCP memory system"""
        if not self._check_connection() or not self.connection:
            return
            
        cursor: Cursor = self.connection.cursor()
        
        # Check if we need to migrate the schema
        self._migrate_schema(cursor)
        
        # Projects table - tracks different coding projects
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT  -- JSON metadata
        )
        """)
        
        # Memories table - stores contextual information and learning
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            type TEXT NOT NULL,  -- 'code', 'conversation', 'decision', 'pattern', 'error'
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding_vector TEXT,  -- JSON array for semantic search
            content_hash TEXT,  -- Hash for duplicate detection
            file_path TEXT,  -- Associated file path
            importance_score REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accessed_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP,
            tags TEXT,  -- JSON array of tags
            metadata TEXT,  -- JSON metadata
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
        """)
        
        # Tasks table - tracks development tasks and TODOs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',  -- 'pending', 'in_progress', 'completed', 'cancelled'
            priority TEXT DEFAULT 'medium',  -- 'low', 'medium', 'high', 'critical'
            category TEXT,  -- 'bug', 'feature', 'refactor', 'docs', 'test'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            due_date TIMESTAMP,
            estimated_hours REAL,
            actual_hours REAL,
            tags TEXT,  -- JSON array
            metadata TEXT,  -- JSON metadata
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
        """)
        
        # Knowledge Graph - relationships between memories, tasks, and concepts
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_relationships (
            id TEXT PRIMARY KEY,
            from_type TEXT NOT NULL,  -- 'memory', 'task', 'project', 'concept'
            from_id TEXT NOT NULL,
            to_type TEXT NOT NULL,
            to_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,  -- 'depends_on', 'relates_to', 'conflicts_with', 'implements', 'references'
            strength REAL DEFAULT 1.0,  -- relationship strength (0.0 to 1.0)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT  -- JSON metadata
        )
        """)
        
        # Sessions table - tracks AI interaction sessions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            interaction_count INTEGER DEFAULT 0,
            context_summary TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
        """)
        
        # Context layers - different context layers for memory retrieval
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS context_layers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            query_pattern TEXT,  -- Pattern for automatic activation
            priority INTEGER DEFAULT 1,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create comprehensive indexes for better performance
        # Memory indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_project_id ON memories(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance_score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_file_path ON memories(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_accessed_count ON memories(accessed_count DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_composite ON memories(project_id, type, importance_score DESC)")
        
        # Task indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_category ON tasks(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_composite ON tasks(project_id, status, priority)")
        
        # Knowledge relationship indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_from ON knowledge_relationships(from_type, from_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_to ON knowledge_relationships(to_type, to_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON knowledge_relationships(relationship_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_strength ON knowledge_relationships(strength DESC)")
        
        # Session indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_project_id ON sessions(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at DESC)")
        
        # Context layer indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_layers_active ON context_layers(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_layers_priority ON context_layers(priority DESC)")
        
        # Notifications table - stores system notifications
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            type TEXT NOT NULL,  -- 'system', 'task', 'memory', 'alert'
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            is_read BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,  -- JSON metadata
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
        """)
        
        # Create indexes for notifications
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notifications_project ON notifications(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(is_read)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at)")
        
        self.connection.commit()
        logging.info("Database tables created successfully")
        
    def _migrate_schema(self, cursor: Cursor) -> None:
        """Migrate database schema to add new columns if they don't exist"""
        try:
            # Check if content_hash column exists in memories table
            cursor.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'content_hash' not in columns:
                cursor.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT")
                logging.info("Added content_hash column to memories table")
                
            if 'file_path' not in columns:
                cursor.execute("ALTER TABLE memories ADD COLUMN file_path TEXT")
                logging.info("Added file_path column to memories table")
                
            if 'embedding_vector' not in columns:
                cursor.execute("ALTER TABLE memories ADD COLUMN embedding_vector TEXT")
                logging.info("Added embedding_vector column to memories table")
                
            # Update accessed_count to INTEGER if it's not already
            if 'accessed_count' in columns:
                # Check current type and update if needed
                cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'")
                table_sql = cursor.fetchone()[0]
                if 'accessed_count REAL' in table_sql or 'accessed_count TEXT' in table_sql:
                    # Create a temporary table with correct schema
                    cursor.execute("""CREATE TABLE memories_temp AS 
                                     SELECT id, project_id, type, title, content, 
                                            CAST(accessed_count AS INTEGER) as accessed_count,
                                            importance_score, created_at, updated_at, 
                                            last_accessed, tags, metadata
                                     FROM memories""")
                    cursor.execute("DROP TABLE memories")
                    cursor.execute("ALTER TABLE memories_temp RENAME TO memories")
                    logging.info("Updated accessed_count column type to INTEGER")
                    
        except Exception as e:
            logging.warning(f"Schema migration warning: {e}")
            # Continue anyway - the table creation will handle missing tables
        
    def get_or_create_project(self, name: str, path: Optional[str] = None, 
                            description: Optional[str] = None) -> str:
        """Get existing project or create new one"""
        if not self._check_connection() or not self.connection:
            return ""
            
        cursor: Cursor = self.connection.cursor()
        
        # Try to find existing project by name or path
        if path:
            cursor.execute("SELECT id FROM projects WHERE path = ? OR name = ?", (path, name))
        else:
            cursor.execute("SELECT id FROM projects WHERE name = ?", (name,))
            
        result = cursor.fetchone()
        if result:
            return result['id']
            
        # Create new project
        project_id = str(uuid.uuid4())
        cursor.execute("""
        INSERT INTO projects (id, name, path, description)
        VALUES (?, ?, ?, ?)
        """, (project_id, name, path, description))
        self.connection.commit()
        return project_id
        
    @retry_on_failure()
    def add_memory(self, project_id: str, memory_type: str, title: str, content: str,
                   tags: Optional[List[str]] = None, importance_score: float = 0.5,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new memory to the database"""
        if not self._check_connection() or not self.connection:
            return ""
            
        cursor: Cursor = self.connection.cursor()
        memory_id = str(uuid.uuid4())
        
        cursor.execute("""
        INSERT INTO memories (id, project_id, type, title, content, importance_score, tags, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, project_id, memory_type, title, content,
            importance_score, json.dumps(tags or []), json.dumps(metadata or {})
        ))
        self.connection.commit()
        return memory_id
        
    @retry_on_failure()
    def get_memories(self, project_id: Optional[str] = None, memory_type: Optional[str] = None,
                     limit: int = 50, sort_by: str = "created_at") -> List[Dict[str, Any]]:
        """Retrieve memories with optional filters"""
        if not self._check_connection() or not self.connection:
            return []
            
        cursor: Cursor = self.connection.cursor()
        
        query = "SELECT * FROM memories WHERE 1=1"
        params = []
        
        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
            
        if memory_type:
            query += " AND type = ?"
            params.append(memory_type)
            
        query += f" ORDER BY {sort_by} DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
        
    @retry_on_failure()
    def add_task(self, project_id: str, title: str, description: Optional[str] = None,
                 priority: str = "medium", category: Optional[str] = None,
                 tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new task"""
        if not self._check_connection() or not self.connection:
            return ""
            
        cursor: Cursor = self.connection.cursor()
        task_id = str(uuid.uuid4())
        
        cursor.execute("""
        INSERT INTO tasks (id, project_id, title, description, priority, category, tags, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_id, project_id, title, description, priority, category,
            json.dumps(tags or []), json.dumps(metadata or {})
        ))
        self.connection.commit()
        return task_id
        
    @retry_on_failure()
    def get_tasks(self, project_id: Optional[str] = None, status: Optional[str] = None,
                  limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve tasks with optional filters"""
        if not self._check_connection() or not self.connection:
            return []
            
        cursor: Cursor = self.connection.cursor()
        
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []
        
        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
            
        if status:
            query += " AND status = ?"
            params.append(status)
            
        query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
        
    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status"""
        if not self._check_connection() or not self.connection:
            return False
            
        cursor: Cursor = self.connection.cursor()
        cursor.execute("""
        UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """, (status, task_id))
        self.connection.commit()
        return cursor.rowcount > 0
        
    def add_relationship(self, from_type: str, from_id: str, to_type: str,
                        to_id: str, relationship_type: str, strength: float = 1.0) -> str:
        """Add a relationship in the knowledge graph"""
        if not self._check_connection() or not self.connection:
            return ""
            
        cursor: Cursor = self.connection.cursor()
        rel_id = str(uuid.uuid4())
        
        cursor.execute("""
        INSERT INTO knowledge_relationships 
        (id, from_type, from_id, to_type, to_id, relationship_type, strength)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (rel_id, from_type, from_id, to_type, to_id, relationship_type, strength))
        self.connection.commit()
        return rel_id
        
    def get_related_items(self, item_type: str, item_id: str) -> List[Dict[str, Any]]:
        """Get all items related to a specific item"""
        if not self._check_connection() or not self.connection:
            return []
            
        cursor: Cursor = self.connection.cursor()
        cursor.execute("""
        SELECT * FROM knowledge_relationships 
        WHERE (from_type = ? AND from_id = ?) OR (to_type = ? AND to_id = ?)
        ORDER BY strength DESC
        """, (item_type, item_id, item_type, item_id))
        return [dict(row) for row in cursor.fetchall()]
        
    def search_memories(self, query: str, project_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Simple text search in memories"""
        if not self._check_connection() or not self.connection:
            return []
            
        cursor: Cursor = self.connection.cursor()
        
        search_query = """
        SELECT * FROM memories 
        WHERE (title LIKE ? OR content LIKE ?)
        """
        params = [f"%{query}%", f"%{query}%"]
        
        if project_id:
            search_query += " AND project_id = ?"
            params.append(project_id)
            
        search_query += " ORDER BY importance_score DESC, created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(search_query, params)
        return [dict(row) for row in cursor.fetchall()]
        
    def update_memory_access(self, memory_id: str) -> None:
        """Update memory access count and timestamp (handles string count)"""
        if not self._check_connection() or not self.connection:
            return
            
        try:
            cursor: Cursor = self.connection.cursor()
            if not cursor:
                return
            
            # First get current count as string
            cursor.execute("SELECT accessed_count FROM memories WHERE id = ?", (memory_id,))
            result = cursor.fetchone()
            if not result:
                return
                
            current_count = int(result['accessed_count']) if result['accessed_count'] else 0
            new_count = str(current_count + 1)
            
            # Update with new string count
            cursor.execute("""
            UPDATE memories 
            SET accessed_count = ?, last_accessed = CURRENT_TIMESTAMP
            WHERE id = ?
            """, (new_count, memory_id))
            self.connection.commit()
        except Exception as e:
            logging.error(f"Error updating memory access: {e}")
        
    def get_project_summary(self, project_id: str) -> Dict[str, Union[Dict[str, str], str]]:
        """Get a summary of project statistics with all string values"""
        if not self._check_connection() or not self.connection:
            return {
                "project": {},
                "memory_counts": {},
                "task_counts": {},
                "total_memories": "0",
                "total_tasks": "0"
            }
            
        cursor: Cursor = self.connection.cursor()
        if not cursor:
            return {
                "project": {},
                "memory_counts": {},
                "task_counts": {},
                "total_memories": "0",
                "total_tasks": "0"
            }

        # Get project info - convert all values to strings
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        project = {str(k): str(v) if v is not None else "" 
                  for k, v in dict(cursor.fetchone() or {}).items()}

        # Get memory counts - cast counts to TEXT in SQL
        cursor.execute("""
            SELECT type, CAST(COUNT(*) AS TEXT) as count FROM memories 
            WHERE project_id = ? GROUP BY type
            """, (project_id,))
        memory_counts = {str(row['type']): str(row['count']) for row in cursor.fetchall()}

        # Get task counts - cast counts to TEXT in SQL
        cursor.execute("""
            SELECT status, CAST(COUNT(*) AS TEXT) as count FROM tasks 
            WHERE project_id = ? GROUP BY status
            """, (project_id,))
        task_counts = {str(row['status']): str(row['count']) for row in cursor.fetchall()}

        # Calculate totals using string values
        total_memories = str(sum(int(count) for count in memory_counts.values()))
        total_tasks = str(sum(int(count) for count in task_counts.values()))

        return {
            "project": project,
            "memory_counts": memory_counts,
            "task_counts": task_counts,
            "total_memories": total_memories,
            "total_tasks": total_tasks
        }
        
    def close(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None

    # ==================== NOTIFICATION METHODS ====================

    def add_notification(self, project_id: str, notification_type: str, 
                        title: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new notification"""
        if not self._check_connection() or not self.connection:
            return ""
            
        cursor: Cursor = self.connection.cursor()
        notification_id = str(uuid.uuid4())
        
        cursor.execute("""
        INSERT INTO notifications (id, project_id, type, title, message, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            notification_id, project_id, notification_type, title, message,
            json.dumps(metadata or {})
        ))
        self.connection.commit()
        return notification_id
        
    def get_notifications(self, project_id: Optional[str] = None, is_read: Optional[bool] = None,
                         limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve notifications with optional filters"""
        if not self._check_connection() or not self.connection:
            return []
            
        cursor: Cursor = self.connection.cursor()
        
        query = "SELECT * FROM notifications WHERE 1=1"
        params = []
        
        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
            
        if is_read is not None:
            query += " AND is_read = ?"
            params.append(int(is_read))
            
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
        
    def mark_notification_read(self, notification_id: str) -> bool:
        """Mark a notification as read"""
        if not self._check_connection() or not self.connection:
            return False
            
        cursor: Cursor = self.connection.cursor()
        cursor.execute("""
        UPDATE notifications SET is_read = 1 WHERE id = ?
        """, (notification_id,))
        self.connection.commit()
        return cursor.rowcount > 0
        
    def clear_notifications(self, project_id: Optional[str] = None) -> int:
        """Clear notifications (optionally for a specific project)"""
        if not self._check_connection() or not self.connection:
            return 0
            
        cursor: Cursor = self.connection.cursor()
        if project_id:
            cursor.execute("DELETE FROM notifications WHERE project_id = ?", (project_id,))
        else:
            cursor.execute("DELETE FROM notifications")
        self.connection.commit()
        return cursor.rowcount

    # ==================== CLEANUP AND OPTIMIZATION METHODS ====================

    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old memories, logs, and completed tasks"""
        if not self._check_connection() or not self.connection:
            return {"memories_deleted": 0, "tasks_deleted": 0, "notifications_deleted": 0}
            
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cursor: Cursor = self.connection.cursor()
            
            # Clean old memories with low importance
            cursor.execute("""
                DELETE FROM memories 
                WHERE created_at < ? AND importance_score < 0.3
            """, (cutoff_date,))
            memories_deleted = cursor.rowcount
            
            # Clean completed tasks older than cutoff
            cursor.execute("""
                DELETE FROM tasks 
                WHERE status = 'completed' AND updated_at < ?
            """, (cutoff_date,))
            tasks_deleted = cursor.rowcount
            
            # Clean old read notifications
            cursor.execute("""
                DELETE FROM notifications 
                WHERE is_read = 1 AND created_at < ?
            """, (cutoff_date,))
            notifications_deleted = cursor.rowcount
            
            self.connection.commit()
            
            return {
                "memories_deleted": memories_deleted,
                "tasks_deleted": tasks_deleted,
                "notifications_deleted": notifications_deleted
            }
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            return {"memories_deleted": 0, "tasks_deleted": 0, "notifications_deleted": 0}

    def optimize_memories(self) -> Dict[str, int]:
        """Analyze and optimize memory storage by removing duplicates"""
        if not self._check_connection() or not self.connection:
            return {"duplicates_merged": 0, "orphaned_relationships": 0}
            
        try:
            cursor: Cursor = self.connection.cursor()
            
            # Find duplicate memories (same content)
            cursor.execute("""
                SELECT content, COUNT(*) as count, GROUP_CONCAT(id) as ids
                FROM memories 
                GROUP BY content 
                HAVING count > 1
            """)
            
            duplicates = cursor.fetchall()
            merged_count = 0
            
            for dup in duplicates:
                ids = dup['ids'].split(',')
                if len(ids) > 1:
                    # Keep the most recent, delete others
                    ids_to_delete = ids[1:]  # Keep first (most recent due to ORDER BY)
                    placeholders = ','.join(['?'] * len(ids_to_delete))
                    cursor.execute(f"""
                        DELETE FROM memories 
                        WHERE id IN ({placeholders})
                    """, ids_to_delete)
                    merged_count += cursor.rowcount
            
            # Clean up orphaned relationships
            cursor.execute("""
                DELETE FROM knowledge_relationships 
                WHERE (from_type = 'memory' AND from_id NOT IN (SELECT id FROM memories))
                   OR (to_type = 'memory' AND to_id NOT IN (SELECT id FROM memories))
                   OR (from_type = 'task' AND from_id NOT IN (SELECT id FROM tasks))
                   OR (to_type = 'task' AND to_id NOT IN (SELECT id FROM tasks))
            """)
            orphaned_relationships = cursor.rowcount
            
            self.connection.commit()
            
            return {
                "duplicates_merged": merged_count,
                "orphaned_relationships": orphaned_relationships
            }
        except Exception as e:
            logging.error(f"Error during optimization: {e}")
            return {"duplicates_merged": 0, "orphaned_relationships": 0}

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        if not self._check_connection() or not self.connection:
            return {}
            
        try:
            cursor: Cursor = self.connection.cursor()
            stats = {}
            
            # Table counts
            tables = ('projects', 'memories', 'tasks', 'knowledge_relationships', 'sessions', 'notifications')
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()['count']
            
            # Database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            stats['database_size_bytes'] = cursor.fetchone()['size']
            
            # Memory statistics
            cursor.execute("""
                SELECT 
                    AVG(importance_score) as avg_importance,
                    MAX(accessed_count) as max_accessed,
                    COUNT(DISTINCT type) as memory_types
                FROM memories
            """)
            memory_stats = cursor.fetchone()
            if memory_stats:
                stats.update({
                    'avg_memory_importance': memory_stats['avg_importance'] or 0,
                    'max_memory_accessed': memory_stats['max_accessed'] or 0,
                    'memory_types_count': memory_stats['memory_types'] or 0
                })
            
            # Task statistics
            cursor.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM tasks
                GROUP BY status
            """)
            task_stats = {row['status']: row['count'] for row in cursor.fetchall()}
            stats['task_status_breakdown'] = task_stats
            
            return stats
        except Exception as e:
            logging.error(f"Error getting database stats: {e}")
            return {}
