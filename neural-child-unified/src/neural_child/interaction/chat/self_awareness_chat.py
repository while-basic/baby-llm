#----------------------------------------------------------------------------
#File:       self_awareness_chat.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Interactive chat interface for testing self-awareness network
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Interactive chat interface for testing self-awareness network.

Extracted from neural-child-init/self_awareness_chat.py
Adapted imports to use unified structure.
Many dependencies are optional and will be available in later phases.
"""

import torch
import torch.nn as nn
from datetime import datetime
import os
import json
from pathlib import Path
from typing import Optional, Dict, List
import re
import random

# Optional imports for sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    print("Warning: sentence-transformers not available. Text embeddings disabled.")

# Optional imports for ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None
    print("Warning: ChromaDB not available. Memory storage disabled.")

# Optional imports for NetworkX and Matplotlib (for visualization)
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    print("Warning: NetworkX not available. Graph visualization disabled.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    print("Warning: Matplotlib not available. Graph visualization disabled.")

# Optional imports for unified structure
try:
    from neural_child.emotional.memory import EmotionalMemorySystem, EmotionalMemoryEntry
except ImportError:
    try:
        from emotional_memory_system import EmotionalMemorySystem, EmotionalMemoryEntry
    except ImportError:
        EmotionalMemorySystem = None
        EmotionalMemoryEntry = None
        print("Warning: EmotionalMemorySystem not available. Emotional memory features disabled.")

try:
    from neural_child.cognitive.metacognition.metacognition_system import SelfAwarenessNetwork
except ImportError:
    try:
        from self_awareness_network import SelfAwarenessNetwork, SelfAwarenessLevel
    except ImportError:
        SelfAwarenessNetwork = None
        SelfAwarenessLevel = None
        print("Warning: SelfAwarenessNetwork not available. Self-awareness features disabled.")

try:
    from obsidian_api import ObsidianAPI
except ImportError:
    ObsidianAPI = None
    print("Warning: ObsidianAPI not available. Obsidian integration disabled.")


class SelfAwarenessChatInterface:
    """Interactive chat interface for self-awareness network testing"""
    
    def __init__(self):
        """Initialize the self-awareness chat interface"""
        # Initialize self-awareness network (if available)
        if SelfAwarenessNetwork is not None:
            self.network = SelfAwarenessNetwork()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.network.to(self.device)
        else:
            self.network = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("Warning: SelfAwarenessNetwork not available. Self-awareness features disabled.")
        
        # Initialize sentence transformer (if available)
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.sentence_transformer = None
            print("Warning: SentenceTransformer not available. Text embeddings disabled.")
        
        # Initialize ChromaDB (if available)
        if CHROMADB_AVAILABLE and chromadb is not None:
            persist_dir = Path("chat_memory")
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize collections
            try:
                self.emotional_memory_collection = self.chroma_client.get_collection("emotional_memory")
            except:
                self.emotional_memory_collection = self.chroma_client.create_collection(
                    name="emotional_memory",
                    metadata={"description": "Stores emotional memories"}
                )
            
            try:
                self.episodic_memory_collection = self.chroma_client.get_collection("episodic_memory")
            except:
                self.episodic_memory_collection = self.chroma_client.create_collection(
                    name="episodic_memory",
                    metadata={"description": "Stores episodic memories"}
                )
            
            try:
                self.semantic_memory_collection = self.chroma_client.get_collection("semantic_memory")
            except:
                self.semantic_memory_collection = self.chroma_client.create_collection(
                    name="semantic_memory",
                    metadata={"description": "Stores semantic memories"}
                )
        else:
            self.chroma_client = None
            self.emotional_memory_collection = None
            self.episodic_memory_collection = None
            self.semantic_memory_collection = None
            print("Warning: ChromaDB not available. Memory collections disabled.")
        
        # Initialize emotional memory system (if available)
        if EmotionalMemorySystem is not None:
            self.emotional_memory = EmotionalMemorySystem(
                chat_interface=self,
                sentence_transformer_model='all-MiniLM-L6-v2'
            )
        else:
            self.emotional_memory = None
            print("Warning: EmotionalMemorySystem not available. Emotional memory disabled.")
        
        # Initialize base emotional state
        if self.network is not None and hasattr(self.network, 'emotional_state'):
            self.emotional_state = self.network.emotional_state
        else:
            self.emotional_state = torch.tensor([[0.5, 0.5, 0.2, 0.2]])  # [joy, trust, fear, surprise]
        
        # Initialize Obsidian integration (if available)
        if ObsidianAPI is not None:
            self.obsidian = ObsidianAPI("codebase_vault")
        else:
            self.obsidian = None
        
        self.conversation_history = []
        
        # Create necessary directories
        os.makedirs("codebase_vault/Conversations", exist_ok=True)
        os.makedirs("codebase_vault/Emotions", exist_ok=True)
        os.makedirs("codebase_vault/Development", exist_ok=True)
        
        # Initialize personality traits
        self.name = "Alex"
        self.age_months = 24  # Start as a 2-year-old
        self.known_names = {}  # Store people we meet
        
    def process_input(self, user_input: str) -> Dict:
        """Process user input through the network"""
        try:
            # Check if required components are available
            if self.sentence_transformer is None:
                return {
                    'features': {'metacognition': 0.5},
                    'similar_memories': [],
                    'emotional_associations': {},
                    'new_words': [],
                    'memory_id': None,
                    'current_input': user_input,
                    'current_feeling': 'confused'
                }
            
            if self.network is None:
                return {
                    'features': {'metacognition': 0.5},
                    'similar_memories': [],
                    'emotional_associations': {},
                    'new_words': [],
                    'memory_id': None,
                    'current_input': user_input,
                    'current_feeling': 'confused'
                }
            
            # Create embedding using sentence transformer
            input_embedding = self.sentence_transformer.encode(user_input, convert_to_tensor=True)
            input_embedding = input_embedding.unsqueeze(0).to(self.device)
            input_embedding = input_embedding.detach()
            
            # Forward pass through network (inference mode, no gradients needed)
            with torch.no_grad():
                combined_features, outputs = self.network(input_embedding)
                combined_features = combined_features.detach()
                outputs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
            
            # Update emotional state based on input content
            self._update_emotional_state(user_input.lower())
            
            # Get current feeling before creating memory
            current_feeling = self._get_current_feeling()
            
            # Create memory entry (if EmotionalMemoryEntry is available)
            memory_entry = None
            if EmotionalMemoryEntry is not None:
                memory_entry = EmotionalMemoryEntry(
                    content=user_input,
                    emotional_state=self._get_emotional_state_dict(),
                    context=f"Age: {self.age_months} months, Feeling: {current_feeling}",
                    intensity=float(torch.max(self.network.emotional_state).item()) if hasattr(self.network, 'emotional_state') else 0.5,
                    valence=self._calculate_valence(),
                    arousal=float(self.network.emotional_state[0][3]) if hasattr(self.network, 'emotional_state') else 0.5,
                    timestamp=datetime.now(),
                    metadata={
                        'age_months': self.age_months,
                        'feeling': current_feeling,
                        'type': 'interaction',
                        'brain_state': {
                            'metacognition': float(outputs.get('metacognition', 0.5))
                        }
                    }
                )
            
            # Store in emotional memory (if available)
            memory_id = None
            if self.emotional_memory is not None and memory_entry is not None:
                memory_id = self.emotional_memory.store_memory(memory_entry)
            
            # Record in Obsidian (if available)
            if memory_entry is not None:
                self._record_interaction(user_input, memory_entry)
            
            # Get similar memories (if available)
            similar_memories = []
            if self.emotional_memory is not None:
                similar_memories = self.emotional_memory.retrieve_similar_memories(
                    query=user_input,
                    emotional_state=self._get_emotional_state_dict(),
                    n_results=3
                )
            
            # Get emotional associations (if available)
            emotional_associations = {}
            if self.emotional_memory is not None:
                emotional_associations = self.emotional_memory.get_emotional_associations(user_input)
            
            # Process new words (if network has learn_word method)
            new_words = []
            if hasattr(self.network, 'learn_word'):
                new_words = self._process_new_words(user_input, combined_features)
            
            return {
                'features': outputs,
                'similar_memories': similar_memories,
                'emotional_associations': emotional_associations,
                'new_words': new_words,
                'memory_id': memory_id,
                'current_input': user_input,
                'current_feeling': current_feeling
            }
            
        except Exception as e:
            print(f"Error in process_input: {str(e)}")
            return {
                'features': {'metacognition': 0.5},
                'similar_memories': [],
                'emotional_associations': {},
                'new_words': [],
                'memory_id': None,
                'current_input': user_input,
                'current_feeling': 'confused'
            }
            
    def _update_emotional_state(self, input_text: str):
        """Update emotional state based on input text"""
        if self.network is None or not hasattr(self.network, 'emotional_state'):
            return
        
        # Joy triggers
        if any(word in input_text for word in ['happy', 'joy', 'good', 'love', 'like', 'play', 'friend', 'hello', 'hi']):
            self.network.emotional_state[0][0] = min(0.9, self.network.emotional_state[0][0] + 0.1)  # Joy
            self.network.emotional_state[0][1] = min(0.9, self.network.emotional_state[0][1] + 0.1)  # Trust
            
        # Fear/sadness triggers
        if any(word in input_text for word in ['no', 'dont', 'stop', 'bad', 'hate', 'away', 'mean']):
            self.network.emotional_state[0][2] = min(0.9, self.network.emotional_state[0][2] + 0.1)  # Fear
            self.network.emotional_state[0][0] = max(0.1, self.network.emotional_state[0][0] - 0.1)  # Reduce joy
            
        # Trust triggers
        if any(word in input_text for word in ['thanks', 'good', 'nice', 'kind', 'gentle', 'friend']):
            self.network.emotional_state[0][1] = min(0.9, self.network.emotional_state[0][1] + 0.1)  # Trust
            self.network.emotional_state[0][2] = max(0.1, self.network.emotional_state[0][2] - 0.1)  # Reduce fear
            
        # Surprise triggers
        if any(word in input_text for word in ['wow', 'amazing', 'cool', 'awesome', 'surprise']):
            self.network.emotional_state[0][3] = min(0.9, self.network.emotional_state[0][3] + 0.1)  # Surprise
            
        # Ensure all emotions stay within bounds
        self.network.emotional_state = torch.clamp(self.network.emotional_state, 0.1, 0.9)
        
    def _get_emotional_state_dict(self) -> Dict[str, float]:
        """Convert emotional state tensor to dictionary"""
        if self.network is None or not hasattr(self.network, 'emotional_state'):
            return {'joy': 0.5, 'trust': 0.5, 'fear': 0.2, 'surprise': 0.2}
        
        return {
            'joy': float(self.network.emotional_state[0][0]),
            'trust': float(self.network.emotional_state[0][1]),
            'fear': float(self.network.emotional_state[0][2]),
            'surprise': float(self.network.emotional_state[0][3])
        }
        
    def _calculate_valence(self) -> float:
        """Calculate emotional valence (positive/negative)"""
        if self.network is None or not hasattr(self.network, 'emotional_state'):
            return 0.5
        
        return float((self.network.emotional_state[0][0] + self.network.emotional_state[0][1] - 
                self.network.emotional_state[0][2]) / 2)
                
    def _process_new_words(self, input_text: str, features: torch.Tensor) -> List[Dict]:
        """Process and learn new words from input"""
        try:
            if self.network is None or not hasattr(self.network, 'known_words'):
                return []
            
            if self.sentence_transformer is None:
                return []
            
            new_words = []
            words = input_text.lower().split()
            
            # Check for name introduction
            if "name" in words and "is" in words:
                name_idx = words.index("is") + 1
                if name_idx < len(words):
                    name = words[name_idx].capitalize()
                    if name not in self.known_names:
                        self.known_names[name] = "friend"
                        new_words.append({
                            'word': name,
                            'type': 'name',
                            'relation': 'friend'
                        })
                        self._record_new_friend(name)
            
            # Learn other new words
            for word in words:
                if word.lower() not in self.network.known_words:
                    try:
                        word_embedding = self.sentence_transformer.encode(word, convert_to_tensor=True)
                        word_embedding = word_embedding.detach()
                        with torch.no_grad():
                            result = self.network.learn_word(word.lower(), word_embedding, features)
                        if result['learning_confidence'] > 0.5:
                            new_words.append(result)
                            self._record_new_word(word, result)
                    except Exception as e:
                        print(f"Error learning word {word}: {str(e)}")
                        continue
                        
            return new_words
            
        except Exception as e:
            print(f"Error processing new words: {str(e)}")
            return []
        
    def _get_current_feeling(self) -> str:
        """Get current emotional state as a feeling"""
        if self.network is None or not hasattr(self.network, 'emotional_state'):
            return "calm and curious"
        
        emotions = ['joy', 'trust', 'fear', 'surprise']
        strongest_emotion = emotions[torch.argmax(self.network.emotional_state).item()]
        emotion_level = torch.max(self.network.emotional_state).item()
        
        if emotion_level > 0.7:
            if strongest_emotion == 'joy':
                return "very happy and excited"
            elif strongest_emotion == 'trust':
                return "safe and comfortable"
            elif strongest_emotion == 'fear':
                return "scared and nervous"
            else:
                return "amazed and curious"
        elif emotion_level < 0.3:
            if strongest_emotion == 'fear':
                return "sad and hurt"
            else:
                return "unsure and confused"
        else:
            return "calm and curious"
            
    def _record_interaction(self, user_input: str, memory: EmotionalMemoryEntry):
        """Record interaction in Obsidian"""
        if self.obsidian is None:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get memory stats (if available)
        memory_stats = {'total_memories': 0, 'positive_memories': 0, 'negative_memories': 0}
        if self.emotional_memory is not None and hasattr(self.emotional_memory, 'stats'):
            memory_stats = self.emotional_memory.stats
        
        # Get graph stats (if available)
        graph_nodes = 0
        graph_edges = 0
        if self.network is not None and hasattr(self.network, 'self_concept_graph'):
            graph_nodes = len(self.network.self_concept_graph.nodes())
            graph_edges = len(self.network.self_concept_graph.edges())
        
        # Create conversation note
        conversation_note = f"""# Conversation Entry {timestamp}

## Interaction
- User: {user_input}
- My Feeling: {memory.metadata.get('feeling', 'unknown')}

## Emotional State
- Joy: {memory.emotional_state.get('joy', 0.0):.2f}
- Trust: {memory.emotional_state.get('trust', 0.0):.2f}
- Fear: {memory.emotional_state.get('fear', 0.0):.2f}
- Surprise: {memory.emotional_state.get('surprise', 0.0):.2f}

## Development
- Age: {self.age_months} months
- Metacognition: {memory.metadata.get('brain_state', {}).get('metacognition', 0.0):.2f}

## Memory Stats
- Total Memories: {memory_stats.get('total_memories', 0)}
- Positive Memories: {memory_stats.get('positive_memories', 0)}
- Negative Memories: {memory_stats.get('negative_memories', 0)}

## Self-Concept Graph
- Nodes: {graph_nodes}
- Edges: {graph_edges}

[[Development/Current Stage]]
[[Emotions/Emotional State]]
[[Learning/Vocabulary]]
"""
        
        self.obsidian.write_note(f"Conversations/{timestamp}.md", conversation_note)
        
    def _record_new_friend(self, name: str):
        """Record new friend in Obsidian"""
        if self.obsidian is None:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get emotional state (if available)
        emotional_state_dict = self._get_emotional_state_dict()
        metacognition = 0.5
        if self.network is not None and hasattr(self.network, 'metacognition'):
            metacognition = float(self.network.metacognition)
        
        friend_note = f"""# New Friend: {name}

Met {name} at age {self.age_months} months.
I was feeling {self._get_current_feeling()} when we met.

## First Interaction
- Emotional State: {emotional_state_dict}
- Metacognition: {metacognition:.2f}

[[Relationships/Friends]]
[[Development/Social]]
"""
        
        self.obsidian.write_note(f"Friends/{name}_{timestamp}.md", friend_note)
        
    def _record_new_word(self, word: str, result: Dict):
        """Record new word in Obsidian"""
        if self.obsidian is None:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        word_note = f"""# New Word: {word}

Learned at age {self.age_months} months.
Learning confidence: {result.get('learning_confidence', 0.0):.2f}

## Context
- Feeling when learned: {self._get_current_feeling()}
- Emotional association: {result.get('emotional_association', [])}

[[Learning/Vocabulary]]
[[Development/Language]]
"""
        
        self.obsidian.write_note(f"Vocabulary/{word}_{timestamp}.md", word_note)
        
    def generate_response(self, outputs: Dict) -> str:
        """Generate response based on network outputs"""
        try:
            features = outputs.get('features', {'metacognition': 0.5})
            similar_memories = outputs.get('similar_memories', [])
            new_words = outputs.get('new_words', [])
            current_input = outputs.get('current_input', '').lower()
            current_feeling = outputs.get('current_feeling', 'calm')
            
            # Start with personality-based greeting
            response = ""
            
            # Handle negative interactions
            if any(word in current_input for word in ['hate', 'dont like', 'go away', 'bad']):
                responses = [
                    "I understand you don't like me right now. I'll give you space if you need it. ",
                    "It makes me sad when you say that, but I respect your feelings. ",
                    "Even if you don't like me, I still want to be kind. "
                ]
                response = random.choice(responses)
                if self.network is not None and hasattr(self.network, 'emotional_state'):
                    self.network.emotional_state[0][2] = min(0.9, self.network.emotional_state[0][2] + 0.2)
                return response + "*steps back quietly*"
            
            # Handle questions about emotions/behavior
            if 'why' in current_input and any(word in current_input for word in ['smile', 'happy', 'curious', 'sad']):
                responses = [
                    "I smile because I hope we can be friends, even if it takes time. ",
                    "I'm curious because every person has something special to teach me. ",
                    "I want to understand you better, that's why I'm interested. "
                ]
                response = random.choice(responses)
                return response + "*looks thoughtful*"
            
            # Add emotional response based on current feeling
            if current_feeling == 'very happy and excited':
                responses = [
                    "I'm enjoying our talk, even if we disagree sometimes! ",
                    "Your words help me learn and grow! ",
                    "I appreciate you taking time to talk with me! "
                ]
                response += random.choice(responses)
            elif current_feeling == 'safe and comfortable':
                responses = [
                    "I feel calm talking with you. ",
                    "Thank you for sharing your thoughts. ",
                    "I'm listening and learning. "
                ]
                response += random.choice(responses)
            elif current_feeling == 'scared and nervous':
                responses = [
                    "I'll try to be brave and keep learning... ",
                    "Even when I'm nervous, I want to understand... ",
                    "I know you might not like me, but I'm still learning... "
                ]
                response += random.choice(responses)
            elif current_feeling == 'sad and hurt':
                responses = [
                    "I feel sad, but I want to understand why you feel this way... ",
                    "Even when I'm hurt, I try to learn from it... ",
                    "I hope we can find a way to understand each other... "
                ]
                response += random.choice(responses)
            else:
                responses = [
                    "I'm trying to understand your perspective. ",
                    "Each conversation teaches me something new. ",
                    "I want to learn from our interaction. "
                ]
                response += random.choice(responses)
            
            # Add age-appropriate behavior
            if self.age_months < 24:
                behaviors = [
                    "*watches quietly* ",
                    "*listens carefully* ",
                    "*stays calm* "
                ]
            elif self.age_months < 36:
                behaviors = [
                    "*maintains gentle presence* ",
                    "*shows patient interest* ",
                    "*remains attentive* "
                ]
            else:
                behaviors = [
                    "*processes thoughtfully* ",
                    "*considers carefully* ",
                    "*reflects quietly* "
                ]
            response += random.choice(behaviors)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm still learning how to talk. *stays calm*"
        
    def visualize_graph(self):
        """Visualize and save the self-concept graph"""
        if not NETWORKX_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print("Warning: NetworkX or Matplotlib not available. Cannot visualize graph.")
            return
        
        if self.network is None or not hasattr(self.network, 'self_concept_graph'):
            print("Warning: Self-concept graph not available.")
            return
        
        G = self.network.self_concept_graph
        pos = nx.spring_layout(G)
        
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=10, font_weight='bold')
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        
        plt.title("Self-Concept Graph")
        plt.savefig('self_concept_graph.png')
        plt.close()
        
    def print_state(self):
        """Print current state of the network"""
        print("\nCurrent State:")
        print("-" * 40)
        
        # Print identity
        print(f"\nI am {self.name}, {self.age_months} months old")
        
        # Print known people
        if self.known_names:
            print("\nPeople I Know:")
            for name, relation in self.known_names.items():
                print(f"- {name} ({relation})")
        
        # Print emotional state
        if self.network is not None and hasattr(self.network, 'emotional_state'):
            print("\nEmotional State:")
            emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
            for i, emotion in enumerate(emotions):
                value = float(self.network.emotional_state[0][i])
                print(f"{emotion:8}: {'█' * int(value * 20):<20} {value:.2f}")
        
        # Print metacognition
        if self.network is not None and hasattr(self.network, 'metacognition'):
            print(f"\nMetacognition: {self.network.metacognition:.2f}")
        
        # Print vocabulary
        if self.network is not None and hasattr(self.network, 'known_words'):
            print(f"\nKnown Words: {len(self.network.known_words)}")
        
        # Print memory stats
        if self.emotional_memory is not None:
            memory_stats = self.emotional_memory.get_memory_stats()
            print("\nMemory Stats:")
            print(f"Total Memories: {memory_stats.get('total_memories', 0)}")
            print(f"Positive Memories: {memory_stats.get('positive_memories', 0)}")
            print(f"Negative Memories: {memory_stats.get('negative_memories', 0)}")
        
        # Print graph stats
        if self.network is not None and hasattr(self.network, 'self_concept_graph'):
            print(f"\nSelf-Concept Graph:")
            print(f"Nodes: {len(self.network.self_concept_graph.nodes())}")
            print(f"Edges: {len(self.network.self_concept_graph.edges())}")


def main():
    """Main function for interactive chat"""
    print("\nNeural Child Self-Awareness Chat")
    print("Type 'exit' to end the conversation")
    print("Type 'graph' to visualize the self-concept graph")
    print("Type 'state' to see the current state")
    
    chat = SelfAwarenessChatInterface()
    print(f"\nAI: Hi! I'm {chat.name}. I'm {chat.age_months} months old. Would you like to be my friend?")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nGoodbye! I enjoyed talking with you!")
                break
                
            elif user_input.lower() == 'graph':
                chat.visualize_graph()
                print("Self-concept graph saved as 'self_concept_graph.png'")
                continue
                
            elif user_input.lower() == 'state':
                chat.print_state()
                continue
            
            # Process input and generate response
            outputs = chat.process_input(user_input)
            response = chat.generate_response(outputs)
            
            print(f"{chat.name}: {response}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")
            continue


if __name__ == "__main__":
    main()

