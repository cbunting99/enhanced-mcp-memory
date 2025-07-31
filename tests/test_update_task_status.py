#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify update_task_status functionality
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import DatabaseManager
from memory_manager import MemoryManager

def test_update_task_status():
    print("=== Testing Update Task Status ===")
    
    # Initialize components
    db_manager = DatabaseManager("data/mcp_memory.db")
    memory_manager = MemoryManager(db_manager)
    
    # Start session
    session_id = memory_manager.start_session()
    print("Started session: {}".format(session_id))
    print("Current project ID: {}".format(memory_manager.current_project_id))
    
    # Create a test task
    task_id = db_manager.add_task(
        project_id=memory_manager.current_project_id,
        title="Test Task for Status Update",
        description="Testing the update_task_status functionality",
        priority="medium",
        category="test"
    )
    print("\n1. Created test task: {}".format(task_id))
    
    # Get initial task status
    tasks = db_manager.get_tasks(memory_manager.current_project_id)
    test_task = next((t for t in tasks if t['id'] == task_id), None)
    print("2. Initial task status: {}".format(test_task['status']))
    
    # Test valid status updates
    valid_statuses = ['in_progress', 'completed', 'cancelled', 'pending']
    
    for status in valid_statuses:
        print("\n3. Updating task to '{}'...".format(status))
        success = db_manager.update_task_status(task_id, status)
        
        if success:
            # Verify the update
            updated_tasks = db_manager.get_tasks(memory_manager.current_project_id)
            updated_task = next((t for t in updated_tasks if t['id'] == task_id), None)
            print("   ✅ Success: Task status is now '{}'".format(updated_task['status']))
        else:
            print("   ❌ Failed to update task status to '{}'".format(status))
    
    # Test invalid status
    print("\n4. Testing invalid status 'invalid_status'...")
    success = db_manager.update_task_status(task_id, 'invalid_status')
    if not success:
        print("   ✅ Correctly rejected invalid status")
    else:
        print("   ❌ Should have rejected invalid status")
    
    # Test non-existent task
    print("\n5. Testing update on non-existent task...")
    success = db_manager.update_task_status('fake-task-id', 'completed')
    if not success:
        print("   ✅ Correctly handled non-existent task")
    else:
        print("   ❌ Should have failed for non-existent task")
    
    # Clean up test task
    print("\n6. Cleaning up test task...")
    cursor = db_manager.connection.cursor()
    cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    db_manager.connection.commit()
    print("   ✅ Test task deleted")
    
    db_manager.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_update_task_status()