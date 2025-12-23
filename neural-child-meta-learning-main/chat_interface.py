import gradio as gr
from typing import Dict, Any
import torch
from datetime import datetime

class NeuralChildInterface:
    def __init__(self, digital_child, mother_llm):
        self.child = digital_child
        self.mother = mother_llm
        self.chat_history = []
        self.current_focus = "mother"  # Toggle between "mother" and "child"
        
    def create_interface(self):
        """Create a Gradio interface for interaction"""
        with gr.Blocks(
            title="Neural Child Development Interface",
            css="h1 { font-family: system-ui, -apple-system, sans-serif; }"
        ) as interface:
            with gr.Row():
                with gr.Column():
                    # Status displays
                    gr.Markdown("# 👶 Neural Child Development System")
                    age_display = gr.Markdown(value=self._get_age_display())
                    emotional_state = gr.Label(
                        label="Emotional State",
                        value={"NEUTRAL": 1.0}
                    )
                    development_stage = gr.Label(
                        label="Development Stage",
                        value={"EARLY_ELEMENTARY": 1.0}
                    )
                
                with gr.Column():
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Interaction Log",
                        height=400,
                        value=[],
                        type="messages"  # Explicitly set message format
                    )
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2
                    )
                    
                    with gr.Row():
                        submit = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear")
                        refresh = gr.Button("Refresh Status")
                        
                    with gr.Row():
                        talk_to_mother = gr.Button("Talk to Mother 👩")
                        talk_to_child = gr.Button("Talk to Child 👶")
            
            # Event handlers
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, emotional_state, development_stage]
            )
            
            submit.click(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, emotional_state, development_stage]
            )
            
            clear.click(
                lambda: ([], {"NEUTRAL": 1.0}, {"EARLY_ELEMENTARY": 1.0}),
                outputs=[chatbot, emotional_state, development_stage]
            )
            
            refresh.click(
                self._update_status,
                outputs=[emotional_state, development_stage]
            )
            
            talk_to_mother.click(
                lambda: self.set_focus("mother"),
                outputs=[]
            )
            
            talk_to_child.click(
                lambda: self.set_focus("child"),
                outputs=[]
            )
            
        return interface
    
    def process_message(self, message: str, history: list) -> tuple:
        """Process incoming messages and generate responses"""
        try:
            if not message.strip():
                return "", history, {"NEUTRAL": 1.0}, {"EARLY_ELEMENTARY": 1.0}
                
            print(f"\nProcessing message: {message}")
            print(f"Current focus: {self.current_focus}")
            
            # Initialize history if None
            history = history or []
            
            try:
                if self.current_focus == "mother":
                    # Check for negative words to amplify negative emotions
                    negative_words = ['suck', 'hate', 'stupid', 'bad', 'dumb', 'ugly', 'mean']
                    is_negative = any(word in message.lower() for word in negative_words)
                    
                    # Convert history for LLM context
                    formatted_history = []
                    for msg in history:
                        formatted_history.append({"role": "user", "content": msg[0]})
                        formatted_history.append({"role": "assistant", "content": msg[1]})
                    
                    # Add current message to context
                    formatted_history.append({"role": "user", "content": message})
                    
                    response = self.mother.respond(message, context=formatted_history)
                    
                    # Update child's emotional state
                    if is_negative:
                        # Amplify fear and reduce trust for negative interactions
                        self.child.update_emotions({
                            'type': 'FEAR',
                            'intensity': 0.8
                        })
                    elif hasattr(self.mother, 'current_emotion'):
                        self.child.update_emotions(self.mother.current_emotion)
                else:
                    response = self.child.process_interaction(message)
                
                if not response:
                    response = "I apologize, but I was unable to generate a response."
                
                # Update history for Gradio chatbot (using list pairs)
                new_history = history + [[message, response]]
                
                # Get updated states
                emotional_state = self.child.get_emotional_state()
                development_stage = self.child.get_development_stage()
                
                # Format emotional state
                if isinstance(emotional_state, dict):
                    if 'state' in emotional_state and 'confidence' in emotional_state:
                        formatted_emotional = {emotional_state['state']: float(emotional_state['confidence'])}
                    else:
                        max_emotion = max(emotional_state.items(), key=lambda x: float(x[1]))
                        formatted_emotional = {max_emotion[0].upper(): float(max_emotion[1])}
                else:
                    formatted_emotional = {"NEUTRAL": 1.0}
                
                # Format development stage
                formatted_development = {str(development_stage): 1.0}
                
                print(f"Emotional state: {formatted_emotional}")
                print(f"Development stage: {formatted_development}")
                
                return "", new_history, formatted_emotional, formatted_development
                
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                error_msg = "I apologize, but I encountered an error while processing your message."
                new_history = history + [[message, error_msg]]
                return "", new_history, {"NEUTRAL": 1.0}, {"EARLY_ELEMENTARY": 1.0}
                
        except Exception as e:
            print(f"Critical error in process_message: {str(e)}")
            return "", history, {"NEUTRAL": 1.0}, {"EARLY_ELEMENTARY": 1.0}
    
    def set_focus(self, target: str):
        """Switch focus between mother and child"""
        self.current_focus = target
    
    def _get_age_display(self) -> str:
        """Get formatted age display"""
        return f"Current Neural Age: {self.child.get_age():.2f} years"
    
    def _update_status(self):
        """Update interface status displays"""
        try:
            emotional_state = self.child.get_emotional_state()
            development_stage = self.child.get_development_stage()
            
            # Format emotional state for Gradio Label
            if isinstance(emotional_state, dict):
                if 'state' in emotional_state and 'confidence' in emotional_state:
                    formatted_emotional = {emotional_state['state']: float(emotional_state['confidence'])}
                else:
                    max_emotion = max(emotional_state.items(), key=lambda x: float(x[1]))
                    formatted_emotional = {max_emotion[0].upper(): float(max_emotion[1])}
            else:
                formatted_emotional = {"NEUTRAL": 1.0}
                
            # Format development stage for Gradio Label
            if isinstance(development_stage, dict):
                if 'stage' in development_stage and 'confidence' in development_stage:
                    formatted_development = {development_stage['stage']: float(development_stage['confidence'])}
                else:
                    formatted_development = {list(development_stage.keys())[0]: 1.0}
            else:
                formatted_development = {str(development_stage): 1.0}
                
            return formatted_emotional, formatted_development
            
        except Exception as e:
            print(f"Error updating status: {str(e)}")
            return {"NEUTRAL": 1.0}, {"EARLY_ELEMENTARY": 1.0}

if __name__ == "__main__":
    from child_model import DynamicNeuralChild
    from main import MotherLLM
    import os
    
    # Initialize the systems
    child = DynamicNeuralChild()
    mother = MotherLLM()
    
    # Create and launch interface
    interface = NeuralChildInterface(child, mother)
    demo = interface.create_interface()
    
    # Configure launch settings
    launch_kwargs = {
        "server_name": "localhost",
        "server_port": 7860,
        "share": False,
        "show_error": True,
        "debug": True,
        "max_threads": 10
    }
    
    # Launch with proper configuration
    demo.queue().launch(**launch_kwargs) 