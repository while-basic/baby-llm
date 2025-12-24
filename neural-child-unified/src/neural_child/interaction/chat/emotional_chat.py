#----------------------------------------------------------------------------
#File:       emotional_chat.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Emotional chat system integrating heartbeat responses and memory recording
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Emotional chat system integrating heartbeat responses and memory recording.

Extracted from neural-child-init/emotional_chat_system.py
Adapted imports to use unified structure.
Many dependencies are optional and will be available in later phases.
"""

import torch
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import json
from pathlib import Path
import re
import numpy as np

# Optional imports for unified structure
try:
    from neural_child.emotional.memory import EmotionalMemoryEntry
except ImportError:
    try:
        from emotional_memory import EmotionalMemoryEntry
    except ImportError:
        EmotionalMemoryEntry = None
        print("Warning: EmotionalMemoryEntry not available. Memory features will be limited.")


class EmotionalChatSystem:
    """Emotional chat system with heartbeat integration and memory recording"""
    
    def __init__(self, brain, obsidian_api=None):
        """Initialize the emotional chat system.
        
        Args:
            brain: IntegratedBrain instance (or compatible object)
            obsidian_api: ObsidianAPI instance (optional)
        """
        self.brain = brain
        self.obsidian_api = obsidian_api
        self.chat_history = []
        self.memory_commands = {
            '!remember': self._handle_remember_command,
            '!forget': self._handle_forget_command,
            '!reflect': self._handle_reflect_command
        }
        
    def process_message(self, message: str, tone: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Process an incoming chat message and update emotional state.
        
        Args:
            message (str): The incoming message
            tone (Dict[str, float], optional): Emotional tone analysis of the message
            
        Returns:
            Dict containing response and state information
        """
        # Check for memory commands
        if message.startswith('!'):
            command = message.split()[0]
            if command in self.memory_commands:
                return self.memory_commands[command](message)
        
        # Process message tone if not provided
        if not tone:
            tone = self._analyze_message_tone(message)
            
        # Update brain's emotional state (if brain has process_emotions method)
        emotional_response = {}
        if hasattr(self.brain, 'process_emotions'):
            try:
                emotional_response = self.brain.process_emotions(
                    torch.tensor([0.0]),
                    torch.tensor([
                        tone.get('joy', 0.0),
                        tone.get('trust', 0.0),
                        tone.get('fear', 0.0),
                        tone.get('surprise', 0.0)
                    ])
                )
            except Exception as e:
                print(f"Warning: Error processing emotions: {str(e)}")
                emotional_response = {}
        
        # Get current heartbeat state (if available)
        heartbeat_info = emotional_response.get('heartbeat', {})
        if hasattr(self.brain, 'heartbeat') and hasattr(self.brain.heartbeat, 'get_current_heartbeat'):
            try:
                heartbeat_info = self.brain.heartbeat.get_current_heartbeat()
            except Exception as e:
                print(f"Warning: Error getting heartbeat: {str(e)}")
        
        # Record interaction in chat history
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'tone': tone,
            'heartbeat': heartbeat_info,
            'brain_state': self.brain.get_brain_state() if hasattr(self.brain, 'get_brain_state') else {}
        }
        self.chat_history.append(chat_entry)
        
        # Record significant emotional interactions in Obsidian (if available)
        if self.obsidian_api is not None and self._is_significant_interaction(tone, heartbeat_info):
            self._record_in_obsidian(chat_entry)
            
        return {
            'emotional_response': emotional_response,
            'heartbeat': heartbeat_info,
            'brain_state': chat_entry['brain_state']
        }
        
    def _analyze_message_tone(self, message: str) -> Dict[str, float]:
        """Analyze the emotional tone of a message.
        
        Args:
            message (str): The message to analyze
            
        Returns:
            Dict containing emotional tone values
        """
        # Simple rule-based tone analysis
        tone = {
            'joy': 0.0,
            'trust': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'anger': 0.0,
            'sadness': 0.0
        }
        
        # Positive indicators
        if re.search(r'happy|great|wonderful|love|excellent|:)', message.lower()):
            tone['joy'] = 0.8
            tone['trust'] = 0.6
            
        # Negative indicators
        if re.search(r'sad|angry|hate|terrible|awful|>:(', message.lower()):
            tone['anger'] = 0.8
            tone['sadness'] = 0.6
            
        # Fear/anxiety indicators
        if re.search(r'scared|afraid|worried|anxious|nervous', message.lower()):
            tone['fear'] = 0.8
            
        # Surprise indicators
        if re.search(r'wow|omg|amazing|incredible|unexpected|\!+', message.lower()):
            tone['surprise'] = 0.8
            
        return tone
        
    def _is_significant_interaction(self, tone: Dict[str, float], 
                                  heartbeat: Dict[str, Any]) -> bool:
        """Determine if an interaction is significant enough to record.
        
        Args:
            tone (Dict[str, float]): Message tone analysis
            heartbeat (Dict[str, Any]): Current heartbeat state
            
        Returns:
            bool: Whether the interaction is significant
        """
        # Check for high emotional values
        high_emotion = any(value > 0.7 for value in tone.values())
        
        # Check for significant heartbeat change
        significant_heartbeat = (
            heartbeat.get('state') in ['ANXIOUS', 'ELEVATED'] or
            abs(heartbeat.get('rate_change', 0)) > 10
        )
        
        return high_emotion or significant_heartbeat
        
    def _record_in_obsidian(self, chat_entry: Dict[str, Any]):
        """Record a significant interaction in Obsidian.
        
        Args:
            chat_entry (Dict[str, Any]): The chat interaction to record
        """
        if self.obsidian_api is None:
            return
        
        # Create markdown content
        timestamp = datetime.fromisoformat(chat_entry['timestamp'])
        filename = f"memory_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        
        content = f"""# Memory Entry - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Interaction
Message: {chat_entry['message']}

## Emotional State
{json.dumps(chat_entry['tone'], indent=2)}

## Heartbeat Response
- Rate: {chat_entry['heartbeat'].get('current_rate', 'N/A')} BPM
- State: {chat_entry['heartbeat'].get('state', 'N/A')}
- Impact: {chat_entry['heartbeat'].get('emotional_impact', 'N/A')}

## Brain State
{json.dumps(chat_entry['brain_state'], indent=2)}

## Tags
#emotional-memory #heartbeat-response"""

        # Save to Obsidian vault
        self.obsidian_api.create_note(filename, content)
        
    def _handle_remember_command(self, message: str) -> Dict[str, Any]:
        """Handle !remember command to store explicit memories.
        
        Args:
            message (str): The full command message
            
        Returns:
            Dict containing command response
        """
        memory_content = message.replace('!remember', '').strip()
        
        # Create memory entry
        timestamp = datetime.now()
        filename = f"explicit_memory_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        
        # Get stage name (if available)
        stage_name = "UNKNOWN"
        if hasattr(self.brain, 'stage') and hasattr(self.brain.stage, 'name'):
            stage_name = self.brain.stage.name
        elif hasattr(self.brain, 'stage'):
            stage_name = str(self.brain.stage)
        
        # Get brain state (if available)
        brain_state = {}
        if hasattr(self.brain, 'get_brain_state'):
            brain_state = self.brain.get_brain_state()
        
        content = f"""# Explicit Memory - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Content
{memory_content}

## Context
- Developmental Stage: {stage_name}
- Brain State: {json.dumps(brain_state, indent=2)}

## Tags
#explicit-memory #commanded-memory"""

        # Save to Obsidian (if available)
        if self.obsidian_api is not None:
            self.obsidian_api.create_note(filename, content)
        
        # Trigger memory processing in brain (if available)
        memory_response = {}
        if hasattr(self.brain, 'process_memory'):
            try:
                memory_response = self.brain.process_memory(
                    torch.tensor([0.0]),
                    memory_type='explicit'
                )
            except Exception as e:
                print(f"Warning: Error processing memory: {str(e)}")
        
        return {
            'status': 'Memory recorded',
            'memory_response': memory_response
        }
        
    def _handle_forget_command(self, message: str) -> Dict[str, Any]:
        """Handle !forget command to mark memories as forgotten.
        
        Args:
            message (str): The full command message
            
        Returns:
            Dict containing command response
        """
        # Implementation for forgetting memories
        # This could involve marking memories in Obsidian with a #forgotten tag
        return {'status': 'Not implemented yet'}
        
    def _handle_reflect_command(self, message: str) -> Dict[str, Any]:
        """Handle !reflect command to analyze emotional patterns.
        
        Args:
            message (str): The full command message
            
        Returns:
            Dict containing reflection analysis
        """
        # Get recent heartbeat history (if available)
        history = []
        if hasattr(self.brain, 'heartbeat') and hasattr(self.brain.heartbeat, 'get_heartbeat_history'):
            try:
                history = self.brain.heartbeat.get_heartbeat_history()
            except Exception as e:
                print(f"Warning: Error getting heartbeat history: {str(e)}")
        
        # Analyze patterns
        analysis = self._analyze_emotional_patterns(history)
        
        # Record reflection in Obsidian (if available)
        if self.obsidian_api is not None:
            timestamp = datetime.now()
            filename = f"reflection_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
            
            content = f"""# Emotional Reflection - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Analysis
{json.dumps(analysis, indent=2)}

## Recent History
{json.dumps(history[-5:], indent=2)}

## Tags
#reflection #emotional-analysis"""

            self.obsidian_api.create_note(filename, content)
        
        return {
            'status': 'Reflection recorded',
            'analysis': analysis
        }
        
    def _analyze_emotional_patterns(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional patterns from heartbeat history.
        
        Args:
            history (List[Dict]): List of heartbeat history entries
            
        Returns:
            Dict containing pattern analysis
        """
        if not history:
            return {'status': 'No history available'}
            
        # Calculate statistics
        rates = [entry['rate'] for entry in history if 'rate' in entry]
        states = [entry['state'] for entry in history if 'state' in entry]
        
        # Calculate emotional stability
        rate_stability = 1.0 - (np.std(rates) / np.mean(rates)) if rates and np.mean(rates) > 0 else 0.0
        
        # Calculate state transitions
        transitions = len([i for i in range(1, len(states)) if states[i] != states[i-1]])
        transition_rate = transitions / len(states) if states else 0.0
        
        # Identify dominant states
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1
            
        dominant_state = max(state_counts.items(), key=lambda x: x[1])[0] if state_counts else None
        
        # Calculate emotional resilience (ability to return to baseline)
        baseline_returns = len([i for i in range(1, len(states)) 
                              if states[i] == 'RESTING' and states[i-1] != 'RESTING'])
        resilience = baseline_returns / transitions if transitions > 0 else 0.0
        
        analysis = {
            'average_rate': np.mean(rates) if rates else 0.0,
            'rate_variance': np.var(rates) if rates else 0.0,
            'rate_stability': rate_stability,
            'dominant_state': dominant_state,
            'state_transitions': transitions,
            'transition_rate': transition_rate,
            'emotional_resilience': resilience,
            'period_analyzed': f"{len(history)} entries",
            'state_distribution': state_counts
        }
        
        return analysis
        
    def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate recent memories and update emotional patterns.
        
        Returns:
            Dict containing consolidation results
        """
        try:
            # Check if emotional_collection is available
            if not hasattr(self, 'emotional_collection') or self.emotional_collection is None:
                return {'status': 'Emotional collection not available'}
            
            # Get recent memories
            recent_memories = self.emotional_collection.get(
                where={"timestamp": {"$gt": (datetime.now() - timedelta(hours=24)).isoformat()}}
            )
            
            if not recent_memories or 'documents' not in recent_memories:
                return {'status': 'No recent memories to consolidate'}
                
            # Group memories by emotional association
            memory_groups = {
                'positive': [],
                'negative': [],
                'neutral': [],
                'complex': [],
                'traumatic': []
            }
            
            for doc, metadata in zip(recent_memories['documents'], recent_memories['metadatas']):
                # Reconstruct emotional state
                emotional_state = {
                    'joy': metadata.get('emotional_joy', 0.0),
                    'trust': metadata.get('emotional_trust', 0.0),
                    'fear': metadata.get('emotional_fear', 0.0),
                    'surprise': metadata.get('emotional_surprise', 0.0)
                }
                
                # Create memory entry (if EmotionalMemoryEntry is available)
                if EmotionalMemoryEntry is not None:
                    memory = EmotionalMemoryEntry(
                        content=doc,
                        emotional_state=emotional_state,
                        context=metadata.get('context', ''),
                        intensity=metadata.get('intensity', 0.0),
                        valence=metadata.get('valence', 0.0),
                        arousal=metadata.get('arousal', 0.0),
                        timestamp=datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat())),
                        metadata=metadata
                    )
                    
                    # Determine association and group
                    association = self._calculate_emotional_association(memory)
                    memory_groups[association.name.lower()].append(memory)
            
            # Calculate consolidation metrics
            total_memories = len(recent_memories['documents'])
            consolidation_metrics = {
                'total_memories': total_memories,
                'positive_ratio': len(memory_groups['positive']) / total_memories if total_memories > 0 else 0.0,
                'negative_ratio': len(memory_groups['negative']) / total_memories if total_memories > 0 else 0.0,
                'emotional_complexity': len(memory_groups['complex']) / total_memories if total_memories > 0 else 0.0,
                'trauma_exposure': len(memory_groups['traumatic']) / total_memories if total_memories > 0 else 0.0
            }
            
            # Update brain's emotional state based on consolidated memories (if available)
            if hasattr(self.brain, 'brain_state'):
                self._update_emotional_state_from_consolidation(memory_groups, consolidation_metrics)
            
            return {
                'status': 'Memory consolidation complete',
                'metrics': consolidation_metrics,
                'memory_distribution': {k: len(v) for k, v in memory_groups.items()}
            }
            
        except Exception as e:
            print(f"Error during memory consolidation: {str(e)}")
            return {'status': 'Error during consolidation', 'error': str(e)}
            
    def _calculate_emotional_association(self, memory) -> Any:
        """Calculate emotional association type for a memory.
        
        Args:
            memory: EmotionalMemoryEntry instance
            
        Returns:
            EmotionalAssociation enum value
        """
        # Simple association calculation
        joy = memory.emotional_state.get('joy', 0.0)
        trust = memory.emotional_state.get('trust', 0.0)
        fear = memory.emotional_state.get('fear', 0.0)
        intensity = memory.intensity
        
        # Determine association type
        if fear > 0.7 and intensity > 0.8:
            return type('EmotionalAssociation', (), {'name': 'TRAUMATIC'})()
        elif (joy + trust) / 2 > 0.5 and fear < 0.3:
            return type('EmotionalAssociation', (), {'name': 'POSITIVE'})()
        elif fear > 0.5:
            return type('EmotionalAssociation', (), {'name': 'NEGATIVE'})()
        elif abs((joy + trust) / 2 - fear) < 0.2:
            return type('EmotionalAssociation', (), {'name': 'NEUTRAL'})()
        else:
            return type('EmotionalAssociation', (), {'name': 'COMPLEX'})()
            
    def _update_emotional_state_from_consolidation(self, 
                                                 memory_groups: Dict[str, List],
                                                 metrics: Dict[str, float]):
        """Update brain's emotional state based on consolidated memories.
        
        Args:
            memory_groups (Dict[str, List]): Grouped memories
            metrics (Dict[str, float]): Consolidation metrics
        """
        try:
            if not hasattr(self.brain, 'brain_state'):
                return
            
            # Calculate base emotional adjustments
            joy_adjustment = metrics['positive_ratio'] - metrics['negative_ratio']
            trust_adjustment = max(0, 0.5 - metrics['trauma_exposure']) * 0.5
            fear_adjustment = metrics['trauma_exposure'] * 0.3
            surprise_adjustment = metrics['emotional_complexity'] * 0.2
            
            # Get current heartbeat info (if available)
            heartbeat_info = {}
            if hasattr(self.brain, 'heartbeat') and hasattr(self.brain.heartbeat, 'get_current_heartbeat'):
                try:
                    heartbeat_info = self.brain.heartbeat.get_current_heartbeat()
                except Exception:
                    pass
            
            current_rate = heartbeat_info.get('rate', 80)
            
            # Calculate heartbeat adjustments
            if current_rate > 100:
                joy_adjustment *= 0.8
                fear_adjustment *= 1.2
            elif current_rate < 70:
                trust_adjustment *= 1.2
                fear_adjustment *= 0.8
                
            # Update brain's emotional state with smoothing (if available)
            if hasattr(self.brain.brain_state, 'emotional_valence'):
                self.brain.brain_state.emotional_valence = max(-1.0, min(1.0,
                    self.brain.brain_state.emotional_valence * 0.7 +
                    (joy_adjustment - fear_adjustment) * 0.3
                ))
            
            if hasattr(self.brain.brain_state, 'arousal'):
                self.brain.brain_state.arousal = max(0.0, min(1.0,
                    self.brain.brain_state.arousal * 0.7 +
                    (surprise_adjustment + fear_adjustment) * 0.3
                ))
            
            # Update neurotransmitter levels (if available)
            if hasattr(self.brain.brain_state, 'neurotransmitters'):
                if 'dopamine' in self.brain.brain_state.neurotransmitters:
                    self.brain.brain_state.neurotransmitters['dopamine'] = max(0.1, min(1.0,
                        self.brain.brain_state.neurotransmitters['dopamine'] * 0.8 +
                        joy_adjustment * 0.2
                    ))
                
                if 'serotonin' in self.brain.brain_state.neurotransmitters:
                    self.brain.brain_state.neurotransmitters['serotonin'] = max(0.1, min(1.0,
                        self.brain.brain_state.neurotransmitters['serotonin'] * 0.8 +
                        trust_adjustment * 0.2
                    ))
                
                if 'norepinephrine' in self.brain.brain_state.neurotransmitters:
                    self.brain.brain_state.neurotransmitters['norepinephrine'] = max(0.1, min(1.0,
                        self.brain.brain_state.neurotransmitters['norepinephrine'] * 0.8 +
                        (fear_adjustment + surprise_adjustment) * 0.2
                    ))
            
        except Exception as e:
            print(f"Error updating emotional state from consolidation: {str(e)}")

