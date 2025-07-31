"""
Automatic Memory Management for MCP Server
Handles context tracking, memory creation, and retrieval
"""

import json
import os
import logging
import sqlite3
import uuid
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from database import DatabaseManager

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic search disabled.")

class MemoryManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.current_session_id = None
        self.current_project_id = None
        self.context_window = []  # Rolling context window
        self.max_context_size = 50
        
        # Initialize embedding model for semantic search
        self.embedding_model = None
        self.embeddings_available = EMBEDDINGS_AVAILABLE
        if self.embeddings_available:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Semantic search enabled with sentence-transformers")
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.embeddings_available = False
        
    def start_session(self, project_path: str = None) -> str:
        """Start a new session, auto-detecting or creating project"""
        project_name = self.detect_project_name(project_path)
        project_description = self.generate_project_description(project_path)
        
        self.current_project_id = self.db.get_or_create_project(
            name=project_name,
            path=project_path,
            description=project_description
        )
        
        # Create new session
        session_uuid = self.generate_id()
        self.db.connection.cursor().execute("""
        INSERT INTO sessions (id, project_id) VALUES (?, ?)
        """, (session_uuid, self.current_project_id))
        self.db.connection.commit()
        
        self.current_session_id = session_uuid
        self.load_relevant_memories()
        
        logging.info(f"Started session for project: {project_name}")
        return session_uuid
        
    def detect_project_name(self, project_path: str = None) -> str:
        """Auto-detect project name from path or workspace"""
        if project_path and os.path.exists(project_path):
            return Path(project_path).name
            
        # Try to detect from current working directory
        cwd = os.getcwd()
        
        # Look for git repository root (walk up the directory tree)
        current_dir = Path(cwd)
        while current_dir != current_dir.parent:
            if (current_dir / '.git').exists():
                return current_dir.name
            current_dir = current_dir.parent
            
        # Check for common project files in current directory
        project_files = ('package.json', 'pyproject.toml', 'Cargo.toml', 'go.mod', 'README.md')
        for file in project_files:
            if os.path.exists(os.path.join(cwd, file)):
                return Path(cwd).name
                
        # If we're in a subdirectory that looks like a project, use parent directory name
        # This helps when running from subdirectories of a project
        parent_indicators = ('src', 'lib', 'tests', 'docs', 'scripts')
        if Path(cwd).name.lower() in parent_indicators:
            return Path(cwd).parent.name
            
        return Path(cwd).name
        
    def generate_project_description(self, project_path: str = None) -> str:
        """Generate project description from available information"""
        if not project_path:
            project_path = os.getcwd()
            
        description_parts = []
        
        # Check for README files
        readme_files = ('README.md', 'README.txt', 'README.rst')
        for readme in readme_files:
            readme_path = os.path.join(project_path, readme)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()[:500]  # First 500 chars
                        description_parts.append(f"README: {content}")
                        break
                except:
                    pass
                    
        # Check for package.json
        package_json = os.path.join(project_path, 'package.json')
        if os.path.exists(package_json):
            try:
                with open(package_json, 'r') as f:
                    data = json.load(f)
                    if 'description' in data:
                        description_parts.append(f"Package description: {data['description']}")
            except:
                pass
                
        return " | ".join(description_parts) if description_parts else "Auto-detected project"
        
    def add_context_memory(self, content: str, memory_type: str = None, importance: float = None, 
                             tags: List[str] = None, file_path: str = None) -> str:
        """Add a new memory to the database with enhanced features"""
        if not self.current_project_id:
            raise ValueError("No active project. Call start_session first.")
        
        # Generate content hash for duplicate detection
        content_hash = self._generate_content_hash(content)
        
        # Check for duplicates
        cursor = self.db.connection.cursor()
        cursor.execute("""
        SELECT id FROM memories WHERE project_id = ? AND content_hash = ?
        """, (self.current_project_id, content_hash))
        
        existing = cursor.fetchone()
        if existing:
            print(f"Duplicate memory detected, updating access count for {existing[0]}")
            cursor.execute("""
            UPDATE memories SET accessed_count = accessed_count + 1, last_accessed = ?
            WHERE id = ?
            """, (datetime.now().isoformat(), existing[0]))
            self.db.connection.commit()
            return existing[0]
        
        # Auto-classify if not provided
        if memory_type is None or importance is None:
            classified_type, classified_importance = self.classify_memory('interaction', content)
            memory_type = memory_type or classified_type
            importance = importance if importance is not None else classified_importance
        
        # Extract title from content
        title = self.extract_title(content)
        
        # Generate embedding for semantic search
        embedding = self._generate_embedding(content)
        embedding_json = json.dumps(embedding) if embedding else None
        
        memory_id = self.generate_id()
        
        cursor.execute("""
        INSERT INTO memories (id, project_id, type, title, content, embedding_vector, 
                            content_hash, file_path, importance_score, tags, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            self.current_project_id,
            memory_type,
            title,
            content,
            embedding_json,
            content_hash,
            file_path,
            importance,
            json.dumps(tags or []),
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        
        self.db.connection.commit()
        
        # Auto-create knowledge relationships
        self._create_auto_relationships(memory_id, content, memory_type)
        
        # Update context window
        self.context_window.append({
            'id': memory_id,
            'type': memory_type,
            'title': title,
            'content': content,
            'importance': importance,
            'file_path': file_path,
            'created_at': datetime.now().isoformat()
        })
        
        # Maintain context window size
        if len(self.context_window) > self.max_context_size:
            self.context_window.pop(0)
        
        return memory_id
        
    def _create_auto_relationships(self, memory_id: str, content: str, memory_type: str):
        """Automatically create relationships between memories"""
        if not self.embedding_model:
            return
            
        try:
            # Get embedding for current memory
            current_embedding = self._generate_embedding(content)
            if not current_embedding:
                return
                
            # Find similar memories
            cursor = self.db.connection.cursor()
            cursor.execute("""
            SELECT id, embedding_vector, title FROM memories 
            WHERE project_id = ? AND id != ? AND embedding_vector IS NOT NULL
            ORDER BY created_at DESC LIMIT 50
            """, (self.current_project_id, memory_id))
            
            similar_memories = []
            for row in cursor.fetchall():
                other_id, embedding_json, title = row
                if embedding_json:
                    try:
                        other_embedding = json.loads(embedding_json)
                        similarity = self._calculate_similarity(current_embedding, other_embedding)
                        if similarity > 0.7:  # High similarity threshold
                            similar_memories.append((other_id, similarity, title))
                    except:
                        continue
                        
            # Create relationships for highly similar memories
            for other_id, similarity, title in similar_memories[:3]:  # Top 3 similar
                rel_id = self.generate_id()
                cursor.execute("""
                INSERT OR IGNORE INTO knowledge_relationships (id, from_type, from_id, to_type, to_id, relationship_type, strength)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (rel_id, 'memory', memory_id, 'memory', other_id, 'similar_content', similarity))
                
            self.db.connection.commit()
            
        except Exception as e:
             print(f"Error creating auto-relationships: {e}")
    
    def search_memories_semantic(self, query: str, limit: int = 10, min_similarity: float = 0.3) -> List[Dict]:
        """Search memories using semantic similarity"""
        if not self.embeddings_available or not self.embedding_model or not self.current_project_id:
            return []
            
        try:
            # Generate embedding for query
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
                
            # Get all memories with embeddings
            cursor = self.db.connection.cursor()
            cursor.execute("""
            SELECT id, title, content, type, importance_score, embedding_vector, file_path, created_at
            FROM memories 
            WHERE project_id = ? AND embedding_vector IS NOT NULL
            ORDER BY created_at DESC
            """, (self.current_project_id,))
            
            results = []
            for row in cursor.fetchall():
                memory_id, title, content, mem_type, importance, embedding_json, file_path, created_at = row
                try:
                    memory_embedding = json.loads(embedding_json)
                    similarity = self._calculate_similarity(query_embedding, memory_embedding)
                    
                    if similarity >= min_similarity:
                        results.append({
                            'id': memory_id,
                            'title': title,
                            'content': content,
                            'type': mem_type,
                            'importance': importance,
                            'file_path': file_path,
                            'created_at': created_at,
                            'similarity': similarity
                        })
                except:
                    continue
                    
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
         
    def classify_memory(self, interaction_type: str, content: str) -> tuple:
        """Enhanced memory classification with better patterns"""
        content_lower = content.lower()
        
        # Enhanced keyword patterns for better classification
        error_keywords = ('error', 'exception', 'failed', 'bug', 'crash', 'traceback', 'stderr', 'warning', 'issue')
        code_keywords = ('function', 'class', 'method', 'import', 'def ', 'async ', 'await', 'return', 'variable', 'algorithm')
        decision_keywords = ('decided', 'choose', 'approach', 'strategy', 'solution', 'implement', 'architecture', 'design')
        pattern_keywords = ('pattern', 'template', 'structure', 'framework', 'convention', 'standard', 'best practice')
        task_keywords = ('todo', 'task', 'need to', 'should', 'must', 'implement', 'fix', 'add', 'create', 'update')
        
        # Determine memory type with priority order
        if any(keyword in content_lower for keyword in error_keywords):
            memory_type = 'error'
            importance = 0.8
        elif any(keyword in content_lower for keyword in task_keywords):
            memory_type = 'task_related'
            importance = 0.7
        elif any(keyword in content_lower for keyword in code_keywords):
            memory_type = 'code'
            importance = 0.7
        elif any(keyword in content_lower for keyword in decision_keywords):
            memory_type = 'decision'
            importance = 0.9
        elif any(keyword in content_lower for keyword in pattern_keywords):
            memory_type = 'pattern'
            importance = 0.6
        else:
            memory_type = 'conversation'
            importance = 0.5
            
        # Enhanced importance scoring
        importance_modifiers = {
            'critical': 0.3, 'important': 0.2, 'urgent': 0.2, 'priority': 0.15,
            'security': 0.25, 'performance': 0.2, 'optimization': 0.15,
            'refactor': 0.1, 'cleanup': 0.05, 'documentation': 0.1
        }
        
        for keyword, modifier in importance_modifiers.items():
            if keyword in content_lower:
                importance += modifier
                
        # Length-based importance adjustment
        if len(content) > 1000:
            importance += 0.15
        elif len(content) > 500:
            importance += 0.1
        elif len(content) < 50:
            importance -= 0.1
            
        # Code complexity indicators
        if memory_type == 'code':
            complexity_indicators = content_lower.count('if ') + content_lower.count('for ') + content_lower.count('while ')
            if complexity_indicators > 3:
                importance += 0.1
                
        return memory_type, min(max(importance, 0.1), 1.0)  # Clamp between 0.1 and 1.0
        
    def extract_title(self, content: str, max_length: int = 100) -> str:
        """Extract meaningful title from content"""
        # Take first sentence or first line
        lines = content.split('\n')
        first_line = lines[0].strip()
        
        if len(first_line) <= max_length:
            return first_line
            
        # Truncate at word boundary
        words = first_line.split()
        title = ""
        for word in words:
            if len(title + word) > max_length:
                break
            title += word + " "
            
        return title.strip() + "..." if title else content[:max_length] + "..."
        
    def load_relevant_memories(self) -> List[Dict]:
        """Load relevant memories for current context"""
        if not self.current_project_id:
            return []
            
        # Get recent high-importance memories
        memories = self.db.get_memories(
            project_id=self.current_project_id,
            limit=20,
            sort_by="importance_score"
        )
        
        # Also get recently accessed memories
        recent_memories = self.db.get_memories(
            project_id=self.current_project_id,
            limit=10,
            sort_by="last_accessed"
        )
        
        # Combine and deduplicate
        all_memories = {m['id']: m for m in memories + recent_memories}
        return list(all_memories.values())
        
    def get_memory_context(self, query: str = None) -> str:
        """Get formatted memory context for AI prompt"""
        if not self.current_project_id:
            return "No active project context."
            
        # Get project summary
        summary = self.db.get_project_summary(self.current_project_id)
        
        # Get relevant memories
        if query:
            memories = self.db.search_memories(query, self.current_project_id, limit=5)
        else:
            memories = self.load_relevant_memories()[:5]
            
        # Get pending tasks
        tasks = self.db.get_tasks(self.current_project_id, status='pending', limit=5)
        
        # Format context
        context_parts = []
        
        # Project summary
        project_info = summary.get('project', {})
        context_parts.append(f"## Current Project: {project_info.get('name', 'Unknown')}")
        if project_info.get('description'):
            context_parts.append(f"Description: {project_info['description']}")
            
        # Memory summary
        memory_counts = summary.get('memory_counts', {})
        if memory_counts:
            context_parts.append(f"Memory Summary: {dict(memory_counts)}")
            
        # Recent memories
        if memories:
            context_parts.append("## Relevant Memories:")
            for memory in memories:
                context_parts.append(f"- [{memory['type']}] {memory['title']}")
                
        # Pending tasks
        if tasks:
            context_parts.append("## Pending Tasks:")
            for task in tasks:
                context_parts.append(f"- [{task['priority']}] {task['title']}")
                
        return "\n".join(context_parts)
        
    def get_task_reminder(self) -> str:
        """Generate task reminder for AI"""
        if not self.current_project_id:
            return ""
            
        pending_tasks = self.db.get_tasks(self.current_project_id, status='pending', limit=3)
        in_progress_tasks = self.db.get_tasks(self.current_project_id, status='in_progress', limit=3)
        
        reminder_parts = []
        
        if pending_tasks or in_progress_tasks:
            reminder_parts.append("## Task Reminder:")
            reminder_parts.append("Remember to create or update tasks for the current project as needed.")
            
            if in_progress_tasks:
                reminder_parts.append("### In Progress:")
                for task in in_progress_tasks:
                    reminder_parts.append(f"- {task['title']} [{task['priority']}]")
                    
            if pending_tasks:
                reminder_parts.append("### Pending:")
                for task in pending_tasks:
                    reminder_parts.append(f"- {task['title']} [{task['priority']}]")
                    
            reminder_parts.append("\nConsider: What tasks need to be added, updated, or completed based on the current conversation?")
            
        return "\n".join(reminder_parts)
        
    def create_project_from_prompt(self, prompt: str) -> str:
        """Create a new project based on user prompt/description"""
        # Extract project name from prompt
        project_name = self.extract_project_name_from_prompt(prompt)
        project_description = prompt[:500]  # Use first 500 chars as description
        
        # Always create a NEW project (don't use get_or_create which might return existing)
        project_id = str(uuid.uuid4())
        cursor = self.db.connection.cursor()
        cursor.execute("""
        INSERT INTO projects (id, name, path, description)
        VALUES (?, ?, ?, ?)
        """, (project_id, project_name, os.getcwd(), project_description))
        self.db.connection.commit()
        
        # Switch to this project
        self.current_project_id = project_id
        self.load_relevant_memories()
        
        logging.info(f"Created new project from prompt: {project_name}")
        return project_id
    
    def extract_project_name_from_prompt(self, prompt: str) -> str:
        """Extract a meaningful project name from user prompt"""
        prompt_lower = prompt.lower()
        
        # Look for explicit project mentions
        project_indicators = [
            r'build (?:a |an )?(.+?)(?:\s+(?:with|using|for|that))',
            r'create (?:a |an )?(.+?)(?:\s+(?:with|using|for|that))',
            r'develop (?:a |an )?(.+?)(?:\s+(?:with|using|for|that))',
            r'make (?:a |an )?(.+?)(?:\s+(?:with|using|for|that))',
            r'implement (?:a |an )?(.+?)(?:\s+(?:with|using|for|that))',
            r'design (?:a |an )?(.+?)(?:\s+(?:with|using|for|that))'
        ]
        
        import re
        for pattern in project_indicators:
            match = re.search(pattern, prompt_lower)
            if match:
                project_name = match.group(1).strip()
                # Clean up the name
                project_name = re.sub(r'[^\w\s-]', '', project_name)
                project_name = ' '.join(project_name.split()[:4])  # Max 4 words
                if len(project_name) > 3:
                    return project_name.title()
        
        # Look for quoted project names
        quoted_match = re.search(r'"([^"]+)"', prompt)
        if quoted_match:
            return quoted_match.group(1)
        
        # Look for framework/technology names
        tech_patterns = [
            r'(\w+)\s+(?:framework|library|api|app|application|system|tool)',
            r'(?:framework|library|api|app|application|system|tool)\s+(?:called|named)\s+(\w+)',
        ]
        
        for pattern in tech_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                return match.group(1).title()
        
        # Fallback: use first few meaningful words
        words = prompt.split()[:6]
        meaningful_words = []
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word and clean_word not in stop_words and len(clean_word) > 2:
                meaningful_words.append(clean_word.title())
                if len(meaningful_words) >= 3:
                    break
        
        if meaningful_words:
            return ' '.join(meaningful_words)
        
        # Ultimate fallback
        return f"Project {datetime.now().strftime('%Y%m%d_%H%M')}"

    def is_new_project_request(self, content: str) -> bool:
        """Check if the content looks like a new project request"""
        content_lower = content.lower()
        
        # Strong indicators of new project requests
        strong_indicators = [
            'build a', 'build an', 'create a', 'create an', 'develop a', 'develop an',
            'make a', 'make an', 'implement a', 'implement an', 'design a', 'design an',
            'start a', 'start an', 'begin a', 'begin an'
        ]
        
        # Project-related keywords
        project_keywords = [
            'framework', 'library', 'api', 'app', 'application', 'system', 'tool',
            'website', 'service', 'platform', 'dashboard', 'interface', 'component',
            'module', 'package', 'project', 'solution', 'prototype'
        ]
        
        # Check for strong indicators + project keywords
        has_strong_indicator = any(indicator in content_lower for indicator in strong_indicators)
        has_project_keyword = any(keyword in content_lower for keyword in project_keywords)
        
        # Additional checks for project scope
        scope_indicators = [
            'with features', 'that can', 'should have', 'requirements', 'specifications',
            'phase', 'roadmap', 'architecture', 'stack', 'technology'
        ]
        has_scope = any(indicator in content_lower for indicator in scope_indicators)
        
        # Length check - project requests are usually detailed
        is_detailed = len(content.split()) > 10
        
        return (has_strong_indicator and has_project_keyword) or (has_strong_indicator and has_scope and is_detailed)

    def auto_create_task_from_context(self, content: str) -> Optional[str]:
        """Enhanced automatic task creation with better detection and project creation"""
        # If no current project, try to create one from the content
        if not self.current_project_id:
            # Check if this looks like a new project request
            if self.is_new_project_request(content):
                self.create_project_from_prompt(content)
            else:
                return None
            
        content_lower = content.lower()
        
        # Enhanced task indicators with patterns
        task_patterns = {
            'action_verbs': ['implement', 'create', 'add', 'build', 'develop', 'write', 'design'],
            'fix_verbs': ['fix', 'resolve', 'debug', 'correct', 'repair', 'patch'],
            'improvement_verbs': ['optimize', 'refactor', 'improve', 'enhance', 'update', 'upgrade'],
            'modal_verbs': ['need to', 'should', 'must', 'have to', 'ought to'],
            'explicit_tasks': ['todo', 'task:', 'action item', 'next step', 'follow up']
        }
        
        # Check for task indicators
        task_score = 0
        detected_category = 'feature'
        detected_priority = 'medium'
        
        for category, patterns in task_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                task_score += 1
                if category == 'fix_verbs':
                    detected_category = 'bug'
                    detected_priority = 'high'
                elif category == 'improvement_verbs':
                    detected_category = 'refactor'
                    
        # Additional context clues
        if any(word in content_lower for word in ['bug', 'error', 'issue', 'problem']):
            detected_category = 'bug'
            detected_priority = 'high'
            task_score += 2
        elif any(word in content_lower for word in ['test', 'testing', 'spec', 'coverage']):
            detected_category = 'test'
        elif any(word in content_lower for word in ['doc', 'documentation', 'readme', 'comment']):
            detected_category = 'docs'
            
        # Priority inference
        if any(word in content_lower for word in ['urgent', 'critical', 'asap', 'immediately']):
            detected_priority = 'critical'
            task_score += 2
        elif any(word in content_lower for word in ['important', 'priority', 'soon']):
            detected_priority = 'high'
            task_score += 1
        elif any(word in content_lower for word in ['minor', 'small', 'quick', 'simple']):
            detected_priority = 'low'
            
        # Require minimum task score to create task
        if task_score < 1:
            return None
            
        # Try to use decompose_task via FastMCP if available
        try:
            from fastmcp import call_tool
            result = call_tool('decompose_task', {'prompt': content})
            if isinstance(result, str):
                return result
        except Exception as e:
            print(f"FastMCP decompose_task failed: {e}")
            
        # Extract meaningful title
        title = self.extract_title(content, max_length=80)
        if not title or len(title) < 10:
            # Try to extract from action patterns
            for pattern in ['implement', 'create', 'add', 'fix', 'update']:
                if pattern in content_lower:
                    start_idx = content_lower.find(pattern)
                    title = content[start_idx:start_idx+80].strip()
                    break
            if not title or len(title) < 10:
                return None
                
        # Create task with enhanced metadata
        task_id = self.db.add_task(
            project_id=self.current_project_id,
            title=title,
            description=content[:500],
            priority=detected_priority,
            category=detected_category,
            tags=['auto-created'],
            metadata={
                'source': 'auto_detection', 
                'created_from': 'conversation',
                'task_score': task_score,
                'auto_enhanced': True
            }
        )
        
        print(f"Auto-created task: {title} (Category: {detected_category}, Priority: {detected_priority})")
        return task_id
        
    def generate_id(self) -> str:
        """Generate unique ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for content to detect duplicates"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text"""
        if not self.embeddings_available or not self.embedding_model:
            return None
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except Exception:
            return 0.0
