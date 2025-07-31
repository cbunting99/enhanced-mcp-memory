"""
Enhanced MCP Server for Memory Management
Automatically manages memories, knowledge graphs, and tasks with improved features
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from functools import wraps
from collections import defaultdict

# Import FastMCP components
try:
    from fastmcp import FastMCP, Context
except ImportError as e:
    logging.error("FastMCP not installed. Install with: pip install fastmcp")
    raise e

from database import DatabaseManager
from memory_manager import MemoryManager

# Configure logging
base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
log_dir = base_dir / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"mcp_memory_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== SERVER CONFIGURATION ====================
class ServerConfig:
    def __init__(self):
        self.max_memory_items = int(os.getenv('MAX_MEMORY_ITEMS', '1000'))
        self.cleanup_interval = int(os.getenv('CLEANUP_INTERVAL_HOURS', '24'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.enable_auto_cleanup = os.getenv('ENABLE_AUTO_CLEANUP', 'true').lower() == 'true'
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', '5'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))

config = ServerConfig()

# ==================== PERFORMANCE TRACKING ====================
class PerformanceTracker:
    def __init__(self):
        self.call_times = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.start_time = datetime.now()
    
    def track_call(self, tool_name: str, duration: float, success: bool = True):
        self.call_times[tool_name].append(duration)
        self.call_counts[tool_name] += 1
        if not success:
            self.error_counts[tool_name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for tool_name, times in self.call_times.items():
            if times:
                stats[tool_name] = {
                    "call_count": self.call_counts[tool_name],
                    "error_count": self.error_counts[tool_name],
                    "avg_time_ms": round(sum(times) / len(times) * 1000, 2),
                    "max_time_ms": round(max(times) * 1000, 2),
                    "min_time_ms": round(min(times) * 1000, 2),
                    "success_rate": round((self.call_counts[tool_name] - self.error_counts[tool_name]) / self.call_counts[tool_name] * 100, 2)
                }
        
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        stats["server_uptime_hours"] = round(uptime_seconds / 3600, 2)
        return stats

perf_tracker = PerformanceTracker()

# Performance tracking decorator
def track_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            perf_tracker.track_call(func.__name__, duration, success)
    return wrapper

# Initialize MCP server
mcp = FastMCP("Enhanced_MCP_Memory")

# Initialize database and memory manager
data_dir = base_dir / "data"
data_dir.mkdir(exist_ok=True)
db_path = data_dir / "mcp_memory.db"

db_manager = DatabaseManager(str(db_path))
memory_manager = MemoryManager(db_manager)

# Load environment variables
load_dotenv()

# ==================== ENHANCED TOOLS ====================

@mcp.tool()
@track_performance
def health_check() -> str:
    """Check server health and database connectivity"""
    try:
        # Test database
        db_manager.connection.execute("SELECT 1").fetchone()
        
        # Test memory manager
        session_count = 1 if memory_manager.current_project_id else 0
        
        # Get basic stats
        stats = db_manager.get_database_stats()
        
        health_info = {
            "status": "healthy",
            "database": "connected",
            "database_size_mb": round(stats.get('database_size_bytes', 0) / (1024*1024), 2),
            "active_sessions": session_count,
            "current_project": memory_manager.current_project_id[:8] + "..." if memory_manager.current_project_id else None,
            "total_projects": stats.get('projects_count', 0),
            "total_memories": stats.get('memories_count', 0),
            "total_tasks": stats.get('tasks_count', 0),
            "server_uptime_hours": perf_tracker.get_stats().get("server_uptime_hours", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(health_info, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

@mcp.tool()
@track_performance
def get_performance_stats() -> str:
    """Get server performance statistics"""
    try:
        stats = perf_tracker.get_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get performance stats: {str(e)}"})

@mcp.tool()
@track_performance
def cleanup_old_data(days_old: int = 30) -> str:
    """Clean up old memories, logs, and completed tasks"""
    try:
        results = db_manager.cleanup_old_data(days_old)
        
        cleanup_summary = {
            "days_threshold": days_old,
            "memories_deleted": results.get("memories_deleted", 0),
            "tasks_deleted": results.get("tasks_deleted", 0),
            "notifications_deleted": results.get("notifications_deleted", 0),
            "cleanup_date": datetime.now().isoformat()
        }
        
        logger.info(f"Cleanup completed: {results}")
        return json.dumps(cleanup_summary, indent=2)
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return json.dumps({"error": f"Cleanup failed: {str(e)}"})

@mcp.tool()
@track_performance
def optimize_memories() -> str:
    """Analyze and optimize memory storage"""
    try:
        results = db_manager.optimize_memories()
        
        optimization_summary = {
            "duplicates_merged": results.get("duplicates_merged", 0),
            "orphaned_relationships_removed": results.get("orphaned_relationships", 0),
            "optimization_complete": True,
            "optimization_date": datetime.now().isoformat()
        }
        
        logger.info(f"Memory optimization completed: {results}")
        return json.dumps(optimization_summary, indent=2)
    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        return json.dumps({"error": f"Memory optimization failed: {str(e)}"})

@mcp.tool()
@track_performance
def get_database_stats() -> str:
    """Get comprehensive database statistics"""
    try:
        stats = db_manager.get_database_stats()
        
        # Add calculated metrics
        if stats.get('memories_count', 0) > 0:
            stats['avg_memories_per_project'] = round(stats['memories_count'] / max(stats.get('projects_count', 1), 1), 2)
        if stats.get('tasks_count', 0) > 0:
            stats['avg_tasks_per_project'] = round(stats['tasks_count'] / max(stats.get('projects_count', 1), 1), 2)
            
        stats['database_size_mb'] = round(stats.get('database_size_bytes', 0) / (1024*1024), 2)
        stats['generated_at'] = datetime.now().isoformat()
        
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get database stats: {str(e)}"})

@mcp.tool()
@track_performance
def get_memory_context(query: str = "") -> str:
    """Get current memory context and task reminders for the AI"""
    try:
        context = memory_manager.get_memory_context(query)
        reminder = memory_manager.get_task_reminder()
        
        full_context = []
        if context:
            full_context.append(context)
        if reminder:
            full_context.append(reminder)
            
        return "\n\n".join(full_context) if full_context else "No context available"
    except Exception as e:
        logger.error(f"Error getting memory context: {e}")
        return f"Error retrieving context: {str(e)}"

@mcp.tool()
@track_performance
def create_task(title: str, description: str = "", priority: str = "medium", category: str = "feature") -> str:
    """Create a new task for the current project"""
    try:
        if not memory_manager.current_project_id:
            memory_manager.start_session()
            
        task_id = db_manager.add_task(
            project_id=memory_manager.current_project_id,
            title=title,
            description=description,
            priority=priority,
            category=category,
            metadata={'source': 'manual', 'created_by': 'ai_tool'}
        )
        
        logger.info(f"Created task: {title} [{priority}/{category}]")
        return f"✅ Task created: '{title}' (ID: {task_id[:8]}...)"
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        return f"⚠️ Error creating task: {str(e)}"

@mcp.tool()
@track_performance
def get_tasks(status: str = None, limit: int = 20) -> str:
    """Get tasks for the current project"""
    try:
        if not memory_manager.current_project_id:
            return "No active project. Cannot retrieve tasks."
            
        tasks = db_manager.get_tasks(memory_manager.current_project_id, status, limit)
        
        results = []
        for task in tasks:
            results.append({
                'id': task['id'],
                'title': task['title'],
                'description': task['description'],
                'status': task['status'],
                'priority': task['priority'],
                'category': task['category'],
                'created_at': task['created_at']
            })
            
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        return f"Error getting tasks: {str(e)}"

@mcp.tool()
@track_performance
def update_task_status(task_id: str, status: str) -> str:
    """Update task status (pending, in_progress, completed, cancelled)"""
    try:
        if not memory_manager.current_project_id:
            return "No active project. Cannot update task."
            
        # Validate status
        valid_statuses = ['pending', 'in_progress', 'completed', 'cancelled']
        if status not in valid_statuses:
            return f"⚠️ Invalid status. Must be one of: {', '.join(valid_statuses)}"
            
        success = db_manager.update_task_status(task_id, status)
        
        if success:
            logger.info(f"Updated task {task_id[:8]}... to status: {status}")
            return f"✅ Task status updated to '{status}'"
        else:
            return f"⚠️ Task not found or update failed"
            
    except Exception as e:
        logger.error(f"Error updating task status: {e}")
        return f"⚠️ Error updating task: {str(e)}"

@mcp.tool()
@track_performance
def get_project_summary() -> str:
    """Get summary of the current project"""
    try:
        if not memory_manager.current_project_id:
            return "No active project."
            
        summary = db_manager.get_project_summary(memory_manager.current_project_id)
        return json.dumps(summary, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error getting project summary: {e}")
        return f"Error getting project summary: {str(e)}"

# ==================== SERVER STARTUP ====================

def initialize_session():
    """Initialize session on server startup"""
    try:
        cwd = os.getcwd()
        memory_manager.start_session(cwd)
        logger.info(f"Session initialized for: {cwd}")
    except Exception as e:
        logger.error(f"Failed to initialize session: {e}")

def main():
    """Main server entry point"""
    try:
        logger.info("🚀 Enhanced MCP Memory Server starting...")
        logger.info(f"📊 Configuration: {config.__dict__}")
        logger.info(f"💾 Database: {db_path}")
        
        # Initialize session
        initialize_session()
        logger.info(f"📁 Current project: {memory_manager.current_project_id}")
        
        # Run the FastMCP server
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        if db_manager:
            db_manager.close()
        logger.info("🛑 Enhanced MCP Memory Server stopped")

if __name__ == "__main__":
    main()