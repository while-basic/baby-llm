#----------------------------------------------------------------------------
#File:       app.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Flask web application for neural child development system
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Flask web application for neural child development system.

Creates a lightweight Flask web interface with RESTful API endpoints
for interacting with the neural child system.
"""

from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
import requests

# Optional imports for unified structure
try:
    from neural_child.visualization.visualization import EmotionalStateVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    EmotionalStateVisualizer = None
    print("Warning: Visualization components not available.")


def create_app(config: Optional[Dict[str, Any]] = None, watch_mode: bool = False) -> Flask:
    """Create and configure the Flask application.
    
    Args:
        config: Optional configuration dictionary
        watch_mode: Enable automatic monitoring features
        
    Returns:
        Configured Flask application
    """
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static'
    )
    
    # Enable CORS
    CORS(app)
    
    # Configuration
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    app.config['JSON_SORT_KEYS'] = False
    app.config['WATCH_MODE'] = watch_mode
    
    if config:
        app.config.update(config)
    
    # Developmental stages progression order
    STAGE_ORDER = [
        'newborn', 'early_infancy', 'late_infancy', 'early_toddler',
        'late_toddler', 'early_preschool', 'late_preschool', 'early_childhood',
        'middle_childhood', 'late_childhood', 'early_elementary',
        'middle_elementary', 'late_elementary', 'early_adolescence',
        'middle_adolescence', 'late_adolescence', 'young_adult', 'mature_adult'
    ]
    
    # Simulated state (will be replaced with actual child model in integration)
    current_state = {
        'emotional_state': {
            'happiness': 0.7,
            'sadness': 0.2,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.3,
            'trust': 0.8,
            'anticipation': 0.6
        },
        'development_speed': 1.0,
        'current_stage': 'infant',
        'age_months': 6.0,
        'warnings': {
            'warning_state': 'normal',
            'metrics': {
                'emotional_stability': 0.8,
                'learning_efficiency': 0.75,
                'attention_level': 0.9,
                'overstimulation_risk': 0.2
            },
            'recent_warnings': []
        }
    }
    
    # Track stage history for progression detection
    stage_history = [current_state['current_stage']]
    last_stage_change = [datetime.now()]  # Use list for mutable reference
    stage_progress = {
        'current_stage_index': 1,  # Index in STAGE_ORDER
        'progress_to_next': 0.45,  # 0.0 to 1.0
        'metrics': {
            'emotional_stability': 0.8,
            'learning_efficiency': 0.75,
            'social_skills': 0.7,
            'cognitive_development': 0.65
        }
    }
    
    # Initialize visualizer if available
    visualizer = None
    if VISUALIZATION_AVAILABLE:
        visualizer = EmotionalStateVisualizer()
    
    # Routes
    @app.route('/')
    def index():
        """Main dashboard page."""
        return render_template('index.html', watch_mode=watch_mode)
    
    @app.route('/api/state', methods=['GET'])
    def get_state():
        """Get current development state.
        
        Returns:
            JSON response with current state
        """
        # Simulate gradual progression (for demo purposes)
        # In real implementation, this would come from actual baby LLM
        import random
        if watch_mode:
            # Simulate small changes over time
            current_state['age_months'] += random.uniform(0.01, 0.05)
            current_state['emotional_state']['happiness'] = max(0.1, min(1.0, 
                current_state['emotional_state']['happiness'] + random.uniform(-0.02, 0.02)))
            
            # Check for stage progression (simulated)
            current_stage_idx = STAGE_ORDER.index(current_state['current_stage']) if current_state['current_stage'] in STAGE_ORDER else 0
            if stage_progress['progress_to_next'] >= 1.0 and current_stage_idx < len(STAGE_ORDER) - 1:
                # Stage progression!
                new_stage = STAGE_ORDER[current_stage_idx + 1]
                if new_stage != current_state['current_stage']:
                    stage_history.append(new_stage)
                    current_state['current_stage'] = new_stage
                    stage_progress['current_stage_index'] = current_stage_idx + 1
                    stage_progress['progress_to_next'] = 0.0
                    last_stage_change[0] = datetime.now()
            else:
                # Gradually increase progress
                stage_progress['progress_to_next'] = min(1.0, 
                    stage_progress['progress_to_next'] + random.uniform(0.001, 0.005))
        
        response = dict(current_state)
        response['watch_mode'] = watch_mode
        response['last_update'] = datetime.now().isoformat()
        return jsonify(response)
    
    @app.route('/api/emotions', methods=['GET'])
    def get_emotions():
        """Get current emotional state.
        
        Returns:
            JSON response with emotional state
        """
        return jsonify(current_state['emotional_state'])
    
    @app.route('/api/emotions', methods=['POST'])
    def update_emotions():
        """Update emotional state.
        
        Request body should contain emotional state values.
        
        Returns:
            JSON response with success status
        """
        data = request.get_json()
        if data:
            current_state['emotional_state'].update(data)
            return jsonify({'status': 'success', 'message': 'Emotional state updated'})
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400
    
    @app.route('/api/memory', methods=['GET'])
    def get_memory():
        """Get memory status.
        
        Returns:
            JSON response with memory information
        """
        # Placeholder - will be replaced with actual memory system
        return jsonify({
            'total_memories': 0,
            'episodic_memories': 0,
            'semantic_memories': 0,
            'emotional_memories': 0
        })
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        """Send a chat message to the neural child.
        
        Request body should contain 'message' field.
        
        Returns:
            JSON response with child's response
        """
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'status': 'error', 'message': 'No message provided'}), 400
        
        message = data['message']
        
        # Try to use actual LLM system
        try:
            from neural_child.interaction.llm.llm_module import chat_completion
            
            # Create a child-appropriate prompt
            stage = current_state.get('current_stage', 'infant')
            age = current_state.get('age_months', 6.0)
            emotions = current_state.get('emotional_state', {})
            
            # Build context-aware prompt
            prompt = f"""You are a {age:.1f}-month-old baby in the {stage} developmental stage.
Your current emotional state:
- Happiness: {emotions.get('happiness', 0.5):.2f}
- Trust: {emotions.get('trust', 0.5):.2f}
- Fear: {emotions.get('fear', 0.1):.2f}
- Surprise: {emotions.get('surprise', 0.3):.2f}

Someone says to you: "{message}"

Respond as a baby would at this age. Keep your response short, simple, and age-appropriate. Use simple words and express your emotions naturally."""
            
            # Get response from Ollama
            llm_response = chat_completion(
                prompt=prompt,
                structured_output=False,
                temperature=0.8,  # Slightly higher for more natural responses
                max_tokens=100  # Keep responses short for a baby
            )
            
            # Extract just the response text if it's a dict
            if isinstance(llm_response, dict):
                response_text = llm_response.get('response', str(llm_response))
            else:
                response_text = str(llm_response)
            
            # Clean up the response (remove any prompt artifacts)
            response_text = response_text.strip()
            if response_text.startswith('"') and response_text.endswith('"'):
                response_text = response_text[1:-1]
            
            # Update emotional state based on interaction (simple simulation)
            import random
            current_state['emotional_state']['happiness'] = min(1.0, 
                current_state['emotional_state']['happiness'] + random.uniform(-0.1, 0.2))
            current_state['emotional_state']['trust'] = min(1.0,
                current_state['emotional_state']['trust'] + random.uniform(-0.05, 0.1))
            
            response = {
                'status': 'success',
                'response': response_text,
                'emotional_state': current_state['emotional_state'],
                'timestamp': datetime.now().isoformat()
            }
            
        except ImportError:
            # Fallback if LLM module not available
            response = {
                'status': 'error',
                'response': "I'm not fully set up yet. Please install Ollama and ensure the LLM module is available.",
                'emotional_state': current_state['emotional_state']
            }
        except requests.exceptions.ConnectionError:
            # Ollama not running
            response = {
                'status': 'error',
                'response': "I can't respond right now. Please make sure Ollama is running. Start it with: ollama serve",
                'emotional_state': current_state['emotional_state'],
                'hint': 'Run "ollama serve" in a terminal to start the Ollama server'
            }
        except Exception as e:
            # Handle any errors gracefully
            error_msg = str(e)
            print(f"Error in chat endpoint: {error_msg}")
            
            # Provide helpful error messages
            if "Connection" in error_msg or "refused" in error_msg.lower():
                response_text = "I can't connect to Ollama. Please make sure Ollama is running (ollama serve)."
            elif "timeout" in error_msg.lower():
                response_text = "The response took too long. Ollama might be busy or the model might not be loaded."
            else:
                response_text = f"I'm having trouble responding. Please check that Ollama is running and the model 'gemma3:1b' is available."
            
            response = {
                'status': 'error',
                'response': response_text,
                'emotional_state': current_state['emotional_state'],
                'error': error_msg
            }
        
        return jsonify(response)
    
    @app.route('/api/visualization/data', methods=['GET'])
    def get_visualization_data():
        """Get visualization data.
        
        Query parameters:
            - type: Type of visualization ('emotional', 'learning', 'network')
            
        Returns:
            JSON response with visualization data
        """
        viz_type = request.args.get('type', 'emotional')
        
        if viz_type == 'emotional' and visualizer:
            data = visualizer.get_emotional_data()
            return jsonify(data)
        elif viz_type == 'learning':
            # Placeholder - will be replaced with actual learning metrics
            return jsonify({
                'loss': [],
                'emotional_stability': [],
                'conversation_quality': []
            })
        elif viz_type == 'network':
            # Placeholder - will be replaced with actual network data
            return jsonify({
                'layers': [],
                'connections': []
            })
        else:
            return jsonify({'error': 'Visualization type not available'}), 404
    
    @app.route('/api/development/speed', methods=['POST'])
    def update_speed():
        """Update development speed.
        
        Request body should contain 'speed' field (float).
        
        Returns:
            JSON response with success status
        """
        data = request.get_json()
        if not data or 'speed' not in data:
            return jsonify({'status': 'error', 'message': 'No speed provided'}), 400
        
        speed = float(data['speed'])
        if speed < 0:
            return jsonify({'status': 'error', 'message': 'Speed cannot be negative'}), 400
        
        current_state['development_speed'] = speed
        return jsonify({'status': 'success', 'message': f'Development speed set to {speed}'})
    
    @app.route('/api/development/warnings', methods=['GET'])
    def get_warnings():
        """Get current warnings.
        
        Returns:
            JSON response with warning information
        """
        return jsonify(current_state['warnings'])
    
    @app.route('/api/neural/activity', methods=['GET'])
    def get_neural_activity():
        """Get neural network activity data.
        
        Returns:
            JSON response with neural activity data
        """
        import random
        import numpy as np
        
        activity = np.random.normal(0.5, 0.15, 10).clip(0, 1).tolist()
        
        return jsonify({
            'timestamp': time.time(),
            'activity_values': activity,
            'mean_activation': float(np.mean(activity)),
            'spike_rate': float(np.random.normal(100, 10)),
            'network_load': float(np.random.normal(0.7, 0.1).clip(0, 1))
        })
    
    @app.route('/api/neural/topology', methods=['GET'])
    def get_neural_topology():
        """Get neural network topology data.
        
        Returns:
            JSON response with network topology
        """
        import random
        import numpy as np
        
        num_nodes = 50
        positions = np.random.normal(0, 1, (num_nodes, 3)).tolist()
        
        num_edges = num_nodes * 2
        edges = []
        for _ in range(num_edges):
            node1 = random.randint(0, num_nodes - 1)
            node2 = random.randint(0, num_nodes - 1)
            if node1 != node2:
                edges.append([node1, node2])
        
        node_activations = np.random.normal(0.5, 0.15, num_nodes).clip(0, 1).tolist()
        edge_weights = np.random.normal(0.5, 0.15, len(edges)).clip(0, 1).tolist()
        
        return jsonify({
            'node_positions': positions,
            'edge_connections': edges,
            'node_activations': node_activations,
            'edge_weights': edge_weights
        })
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint.
        
        Returns:
            JSON response with health status
        """
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    @app.route('/api/progress', methods=['GET'])
    def get_progress():
        """Get developmental progress information.
        
        Returns:
            JSON response with progress data
        """
        current_stage = current_state['current_stage']
        current_idx = STAGE_ORDER.index(current_stage) if current_stage in STAGE_ORDER else 0
        next_stage = STAGE_ORDER[current_idx + 1] if current_idx < len(STAGE_ORDER) - 1 else None
        
        return jsonify({
            'current_stage': current_stage,
            'next_stage': next_stage,
            'progress_to_next': stage_progress['progress_to_next'],
            'stage_index': current_idx,
            'total_stages': len(STAGE_ORDER),
            'metrics': stage_progress['metrics'],
            'stage_history': stage_history[-10:],  # Last 10 stages
            'last_stage_change': last_stage_change[0].isoformat(),
            'time_in_current_stage': (datetime.now() - last_stage_change[0]).total_seconds()
        })
    
    @app.route('/api/stage-changes', methods=['GET'])
    def get_stage_changes():
        """Get recent stage changes for alerts.
        
        Query parameters:
            - since: ISO timestamp to get changes since (optional)
            
        Returns:
            JSON response with stage change events
        """
        since_str = request.args.get('since')
        changes = []
        
        if since_str:
            try:
                since = datetime.fromisoformat(since_str)
                # Return changes since the given time
                # For now, return empty as we're simulating
                pass
            except:
                pass
        
        # Return any new stage changes
        if len(stage_history) > 1:
            changes.append({
                'timestamp': last_stage_change[0].isoformat(),
                'from_stage': stage_history[-2] if len(stage_history) > 1 else None,
                'to_stage': stage_history[-1],
                'message': f"Progressed from {stage_history[-2]} to {stage_history[-1]}"
            })
        
        return jsonify({
            'changes': changes,
            'current_stage': current_state['current_stage']
        })
    
    return app


# Alias for compatibility
app_factory = create_app

