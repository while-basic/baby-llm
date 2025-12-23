"""NeuralChild Interactive Dashboard

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License

Interactive dashboard for real-time visualization and monitoring of NeuralChild
developmental simulation, including neural networks, emotions, and cognitive state.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import threading
import time
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# NeuralChild imports
from neuralchild.config import get_config
from neuralchild.mind.mind_core import Mind
from neuralchild.mother.mother_llm import MotherLLM
from neuralchild.mind.networks.consciousness import ConsciousnessNetwork
from neuralchild.mind.networks.emotions import EmotionsNetwork
from neuralchild.mind.networks.perception import PerceptionNetwork
from neuralchild.mind.networks.thoughts import ThoughtsNetwork
from neuralchild.core.schemas import DevelopmentalStage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DashboardState:
    """Global state management for the dashboard."""

    def __init__(self):
        self.mind: Mind = Mind()
        self.mother: MotherLLM = MotherLLM()
        self.networks: Dict[str, Any] = {}
        self.is_running: bool = False
        self.step_count: int = 0
        self.step_interval: float = 0.1
        self.save_interval: int = 100
        self.last_saved_step: int = 0
        self.save_directory: str = "saved_models"
        self.auto_backup: bool = True

        # History tracking
        self.development_history: List[Dict[str, Any]] = []
        self.emotion_history: List[Dict[str, Any]] = []
        self.memory_history: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.last_mother_response: Optional[str] = None

        # State cache
        self.mind_state: Dict[str, Any] = {}
        self.network_states: Dict[str, Dict[str, Any]] = {}

        self._init_networks()

    def _init_networks(self):
        """Initialize neural networks."""
        try:
            self.networks = {
                "consciousness": ConsciousnessNetwork(),
                "emotions": EmotionsNetwork(),
                "perception": PerceptionNetwork(),
                "thoughts": ThoughtsNetwork()
            }
            for network in self.networks.values():
                self.mind.register_network(network)
            logger.info("Neural networks initialized successfully")
        except Exception as e:
            error_msg = f"Error initializing networks: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)

    def update_state(self):
        """Update the current state from mind."""
        try:
            observable_state = self.mind.get_observable_state()
            mind_state = self.mind.get_state()

            self.mind_state = {
                "developmental_stage": observable_state.developmental_stage.name,
                "energy_level": observable_state.energy_level,
                "apparent_mood": observable_state.apparent_mood,
                "vocalization": observable_state.vocalization or "None",
                "current_focus": observable_state.current_focus or "None",
                "consciousness_level": mind_state.consciousness_level,
                "timestamp": datetime.now().isoformat()
            }

            # Update network states
            for name, network in self.networks.items():
                try:
                    text_output = network.generate_text_output()
                    self.network_states[name] = {
                        "text_output": text_output.text,
                        "confidence": text_output.confidence
                    }
                except Exception as e:
                    logger.warning(f"Error updating network {name}: {str(e)}")

            # Track history
            self._update_history(observable_state, mind_state)

        except Exception as e:
            error_msg = f"Error updating state: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            self._trim_errors()

    def _update_history(self, observable_state, mind_state):
        """Update history tracking."""
        # Development history
        self.development_history.append({
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "developmental_stage": observable_state.developmental_stage.name,
            "energy_level": observable_state.energy_level,
            "consciousness_level": mind_state.consciousness_level
        })
        if len(self.development_history) > 1000:
            self.development_history = self.development_history[-1000:]

        # Emotion history
        if observable_state.recent_emotions:
            for emotion in observable_state.recent_emotions:
                self.emotion_history.append({
                    "step": self.step_count,
                    "timestamp": datetime.now().isoformat(),
                    "emotion": emotion.name.value,
                    "intensity": emotion.intensity
                })
            if len(self.emotion_history) > 1000:
                self.emotion_history = self.emotion_history[-1000:]

        # Memory history
        self.memory_history.append({
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "short_term": len(self.mind.short_term_memory),
            "long_term": len(self.mind.long_term_memory)
        })
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-1000:]

    def _trim_errors(self):
        """Keep only recent errors."""
        if len(self.errors) > 50:
            self.errors = self.errors[-50:]

    def save_checkpoint(self) -> tuple[bool, str]:
        """Save a checkpoint of the current state."""
        try:
            save_path = Path(self.save_directory)
            save_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = save_path / f"checkpoint_{timestamp}_step_{self.step_count}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            self.mind.save_state(str(checkpoint_dir))

            # Save dashboard data
            dashboard_data = {
                "step_count": self.step_count,
                "development_history": self.development_history[-100:],
                "emotion_history": self.emotion_history[-100:],
                "memory_history": self.memory_history[-100:]
            }

            with open(checkpoint_dir / "dashboard_data.json", "w") as f:
                json.dump(dashboard_data, f, indent=2)

            self.last_saved_step = self.step_count
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
            return True, str(checkpoint_dir)

        except Exception as e:
            error_msg = f"Error saving checkpoint: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            self._trim_errors()
            return False, error_msg


# Global state instance
state = DashboardState()
simulation_thread = None
simulation_active = False


def run_simulation():
    """Background simulation loop."""
    global simulation_active, state

    simulation_active = True
    state.is_running = True

    logger.info("Simulation started")

    while simulation_active:
        try:
            # Run simulation step
            state.mind.step()
            state.step_count += 1

            # Update state
            state.update_state()

            # Mother interaction every 10 steps
            if state.step_count % 10 == 0:
                try:
                    response = state.mother.observe_and_respond(state.mind)
                    if response:
                        state.last_mother_response = response.response
                except Exception as e:
                    logger.warning(f"Mother interaction error: {str(e)}")

            # Auto-save checkpoint
            if (state.auto_backup and
                state.step_count % state.save_interval == 0 and
                state.step_count > state.last_saved_step):
                state.save_checkpoint()

            # Sleep for configured interval
            time.sleep(state.step_interval)

        except Exception as e:
            error_msg = f"Simulation error at step {state.step_count}: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state._trim_errors()
            time.sleep(1)

    state.is_running = False
    logger.info("Simulation stopped")


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="NeuralChild Dashboard"
)

# App Layout
app.layout = dbc.Container(
    fluid=True,
    className="p-4",
    children=[
        # Auto-refresh interval
        dcc.Interval(id='refresh-interval', interval=1000, n_intervals=0),
        dcc.Store(id='state-store'),

        # Header
        dbc.Row([
            dbc.Col([html.H1("NeuralChild Dashboard", className="text-primary mb-3"),
                    html.P("Real-time Neural Development Monitoring", className="text-muted")], width=8),
            dbc.Col([html.Div([dbc.Badge(id="status-badge", className="me-2", children="Inactive"),
                              dbc.Badge(id="step-badge", className="me-2", children="Step: 0")])
            ], width=4, className="text-end")
        ], className="mb-4"),

        # Control Panel
        dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("Simulation Controls"), dbc.CardBody([
                        dbc.ButtonGroup([dbc.Button("Start", id="start-btn", color="success", className="me-2"),
                                        dbc.Button("Stop", id="stop-btn", color="danger", className="me-2"),
                                        dbc.Button("Save", id="save-btn", color="info")], className="mb-3"),
                        dbc.Row([
                            dbc.Col([dbc.Label("Step Interval (s)"),
                                    dbc.Input(id="step-interval", type="number", value=0.1, step=0.01, min=0.01)], width=4),
                            dbc.Col([dbc.Label("Save Interval"),
                                    dbc.Input(id="save-interval", type="number", value=100, step=10, min=10)], width=4),
                            dbc.Col([dbc.Checkbox(id="auto-backup", label="Auto-backup", value=True)],
                                   width=4, className="d-flex align-items-end")
                        ])])
                ])], width=12)], className="mb-4"),

        # Main Content
        dbc.Row([
            # Left Column - Mind State
            dbc.Col([
                dbc.Card([dbc.CardHeader("Mind State"), dbc.CardBody([
                        html.H4(id="dev-stage", className="text-warning mb-3"),
                        dbc.Row([
                            dbc.Col([html.Label("Consciousness"), dbc.Progress(id="consciousness-bar", className="mb-2")], width=6),
                            dbc.Col([html.Label("Energy"), dbc.Progress(id="energy-bar", color="success", className="mb-2")], width=6)
                        ]),
                        dbc.Row([dbc.Col([html.Label("Mood"), dbc.Progress(id="mood-bar", color="info", className="mb-2")], width=12)]),
                        html.Hr(),
                        html.Div([html.Strong("Focus: "), html.Span(id="focus-text", className="text-muted")], className="mb-2"),
                        html.Div([html.Strong("Vocalization: "), html.Span(id="vocal-text", className="text-muted")])
                    ])], className="mb-3"),

                dbc.Card([dbc.CardHeader("Mother's Response"), dbc.CardBody([
                        html.Div(id="mother-text", className="text-light fst-italic")])
                ], className="mb-3"),

                dbc.Card([dbc.CardHeader("System Log"), dbc.CardBody([
                        html.Div(id="log-text", className="font-monospace small",
                                style={"maxHeight": "150px", "overflowY": "auto"})])
                ])
            ], width=4),

            # Right Column - Networks & Analytics
            dbc.Col([
                dcc.Tabs(id="main-tabs", value="networks", children=[
                    dcc.Tab(label="Neural Networks", value="networks", children=[
                        dbc.Row([
                            dbc.Col([dbc.Card([dbc.CardHeader("Consciousness"), dbc.CardBody([
                                        html.Div(id="consciousness-text", className="mb-2"),
                                        dbc.Progress(id="consciousness-conf", className="mb-1", style={"height": "5px"})])
                                ], className="mt-3")], width=6),
                            dbc.Col([dbc.Card([dbc.CardHeader("Emotions"), dbc.CardBody([
                                        html.Div(id="emotions-text", className="mb-2"),
                                        dbc.Progress(id="emotions-conf", className="mb-1", style={"height": "5px"})])
                                ], className="mt-3")], width=6)
                        ]),
                        dbc.Row([
                            dbc.Col([dbc.Card([dbc.CardHeader("Perception"), dbc.CardBody([
                                        html.Div(id="perception-text", className="mb-2"),
                                        dbc.Progress(id="perception-conf", className="mb-1", style={"height": "5px"})])
                                ], className="mt-3")], width=6),
                            dbc.Col([dbc.Card([dbc.CardHeader("Thoughts"), dbc.CardBody([
                                        html.Div(id="thoughts-text", className="mb-2"),
                                        dbc.Progress(id="thoughts-conf", className="mb-1", style={"height": "5px"})])
                                ], className="mt-3")], width=6)
                        ])
                    ]),

                    dcc.Tab(label="Analytics", value="analytics", children=[
                        dbc.Card([dbc.CardHeader("Development Over Time"),
                                 dbc.CardBody([dcc.Graph(id="dev-graph", config={"displayModeBar": False})])
                        ], className="mt-3 mb-3"),
                        dbc.Card([dbc.CardHeader("Emotional Evolution"),
                                 dbc.CardBody([dcc.Graph(id="emotion-graph", config={"displayModeBar": False})])
                        ], className="mb-3"),
                        dbc.Card([dbc.CardHeader("Memory Growth"),
                                 dbc.CardBody([dcc.Graph(id="memory-graph", config={"displayModeBar": False})])
                        ])
                    ]),

                    dcc.Tab(label="Development", value="development", children=[
                        dbc.Card([dbc.CardHeader("Developmental Milestones"),
                                 dbc.CardBody([html.Div(id="milestones-div")])
                        ], className="mt-3 mb-3"),
                        dbc.Card([dbc.CardHeader("Beliefs & Needs"),
                                 dbc.CardBody([html.Div(id="beliefs-div")])
                        ])
                    ])
                ])
            ], width=8)
        ])
    ]
)


# Callbacks
@app.callback(
    Output('state-store', 'data'),
    Input('refresh-interval', 'n_intervals')
)
def update_state_store(n):
    """Update the state store with current data."""
    return {
        "is_running": state.is_running,
        "step_count": state.step_count,
        "mind_state": state.mind_state,
        "network_states": state.network_states,
        "last_mother_response": state.last_mother_response,
        "errors": state.errors
    }


@app.callback(
    [Output('status-badge', 'children'),
     Output('status-badge', 'color'),
     Output('step-badge', 'children')],
    Input('state-store', 'data')
)
def update_status(data):
    """Update status indicators."""
    if not data:
        return "Inactive", "secondary", "Step: 0"

    is_running = data.get("is_running", False)
    step_count = data.get("step_count", 0)

    status = "Running" if is_running else "Inactive"
    color = "success" if is_running else "secondary"
    step_text = f"Step: {step_count}"

    return status, color, step_text


@app.callback(
    [Output('dev-stage', 'children'),
     Output('consciousness-bar', 'value'),
     Output('energy-bar', 'value'),
     Output('mood-bar', 'value'),
     Output('focus-text', 'children'),
     Output('vocal-text', 'children'),
     Output('mother-text', 'children')],
    Input('state-store', 'data')
)
def update_mind_state(data):
    """Update mind state display."""
    if not data or not data.get("mind_state"):
        return "INFANT", 20, 70, 50, "None", "None", "No recent responses."

    mind = data["mind_state"]

    stage = mind.get("developmental_stage", "INFANT")
    consciousness = int(mind.get("consciousness_level", 0.2) * 100)
    energy = int(mind.get("energy_level", 0.7) * 100)
    mood = int((mind.get("apparent_mood", 0) + 1) * 50)
    focus = mind.get("current_focus", "None")
    vocal = mind.get("vocalization", "None")
    mother = data.get("last_mother_response", "No recent responses.")

    return stage, consciousness, energy, mood, focus, vocal, mother or "No recent responses."


@app.callback(
    [Output('consciousness-text', 'children'),
     Output('consciousness-conf', 'value'),
     Output('emotions-text', 'children'),
     Output('emotions-conf', 'value'),
     Output('perception-text', 'children'),
     Output('perception-conf', 'value'),
     Output('thoughts-text', 'children'),
     Output('thoughts-conf', 'value')],
    Input('state-store', 'data')
)
def update_networks(data):
    """Update network outputs."""
    if not data or not data.get("network_states"):
        return ("No data", 0) * 4

    networks = data["network_states"]

    outputs = []
    for net_name in ["consciousness", "emotions", "perception", "thoughts"]:
        net = networks.get(net_name, {})
        text = net.get("text_output", "No data")
        conf = int(net.get("confidence", 0) * 100)
        outputs.extend([text, conf])

    return tuple(outputs)


@app.callback(
    Output('dev-graph', 'figure'),
    Input('refresh-interval', 'n_intervals')
)
def update_dev_graph(n):
    """Update development graph."""
    if not state.development_history:
        return create_empty_figure("No development data")

    df = pd.DataFrame(state.development_history[-200:])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["consciousness_level"],
        mode='lines', name='Consciousness',
        line=dict(width=2, color='#6610f2')
    ))
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["energy_level"],
        mode='lines', name='Energy',
        line=dict(width=2, color='#28a745')
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Steps",
        yaxis_title="Level",
        height=250,
        legend=dict(orientation="h", yanchor="top", y=1.1)
    )

    return fig


@app.callback(
    Output('emotion-graph', 'figure'),
    Input('refresh-interval', 'n_intervals')
)
def update_emotion_graph(n):
    """Update emotion graph."""
    if not state.emotion_history:
        return create_empty_figure("No emotion data")

    df = pd.DataFrame(state.emotion_history[-200:])
    df = df.groupby(['step', 'emotion'])['intensity'].max().reset_index()
    df_pivot = df.pivot(index='step', columns='emotion', values='intensity').fillna(0)

    colors = {
        'joy': '#ffc107', 'sadness': '#17a2b8', 'anger': '#dc3545',
        'fear': '#6f42c1', 'surprise': '#fd7e14', 'trust': '#20c997'
    }

    fig = go.Figure()
    for emotion in df_pivot.columns:
        fig.add_trace(go.Scatter(
            x=df_pivot.index, y=df_pivot[emotion],
            mode='lines', name=emotion.capitalize(),
            line=dict(width=2, color=colors.get(emotion, '#adb5bd'))
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Steps",
        yaxis_title="Intensity",
        height=250,
        legend=dict(orientation="h", yanchor="top", y=1.1)
    )

    return fig


@app.callback(
    Output('memory-graph', 'figure'),
    Input('refresh-interval', 'n_intervals')
)
def update_memory_graph(n):
    """Update memory graph."""
    if not state.memory_history:
        return create_empty_figure("No memory data")

    df = pd.DataFrame(state.memory_history[-200:])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["short_term"],
        mode='lines', name='Short-term',
        line=dict(width=2, color='#6610f2')
    ))
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["long_term"],
        mode='lines', name='Long-term',
        line=dict(width=2, color='#007bff')
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Steps",
        yaxis_title="Memory Count",
        height=250,
        legend=dict(orientation="h", yanchor="top", y=1.1)
    )

    return fig


@app.callback(
    Output('milestones-div', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_milestones(n):
    """Update developmental milestones display."""
    try:
        milestones = state.mind.developmental_milestones
        current_stage = state.mind.state.developmental_stage

        content = [
            html.H5(f"Current Stage: {current_stage.name}", className="mb-3"),
        ]

        # Get vocabulary
        vocab = list(milestones.get("vocabulary_learned", set()))
        if vocab:
            vocab_sample = sorted(vocab)[-15:]
            content.extend([
                html.H6("Recent Vocabulary:", className="mt-3"),
                html.P(", ".join(vocab_sample), className="text-muted"),
                html.Small(f"Total: {len(vocab)} words", className="text-secondary")
            ])

        return content

    except Exception as e:
        logger.error(f"Error updating milestones: {str(e)}")
        return html.Div("Error loading milestones")


@app.callback(
    Output('beliefs-div', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_beliefs(n):
    """Update beliefs and needs display."""
    try:
        beliefs = list(state.mind.belief_network.beliefs.values())
        needs = state.mind.need_system.needs

        content = []

        # Beliefs
        if beliefs:
            content.append(html.H6("Recent Beliefs:", className="mb-2"))
            for belief in beliefs[-5:]:
                content.append(
                    html.Div(
                        f"{belief.subject} {belief.predicate} {belief.object} "
                        f"({int(belief.confidence * 100)}%)",
                        className="small text-muted mb-1"
                    )
                )

        # Needs
        if needs:
            content.append(html.H6("Needs Status:", className="mt-3 mb-2"))
            for name, need in list(needs.items())[:5]:
                content.append(
                    html.Div([
                        html.Small(name.capitalize()),
                        dbc.Progress(
                            value=int(need.intensity * 100),
                            color="warning" if need.intensity > 0.7 else "info",
                            className="mb-2",
                            style={"height": "8px"}
                        )
                    ])
                )

        return content if content else html.Div("No data available")

    except Exception as e:
        logger.error(f"Error updating beliefs: {str(e)}")
        return html.Div("Error loading beliefs")


@app.callback(
    Output('log-text', 'children'),
    Input('state-store', 'data')
)
def update_log(data):
    """Update system log."""
    if not data or not data.get("errors"):
        return "System ready."

    errors = data["errors"][-10:]
    return "\n".join(f"[{i+1}] {err}" for i, err in enumerate(errors))


@app.callback(
    Output('refresh-interval', 'n_intervals', allow_duplicate=True),
    Input('start-btn', 'n_clicks'),
    prevent_initial_call=True
)
def start_simulation_callback(n_clicks):
    """Start the simulation."""
    if n_clicks:
        global simulation_thread, simulation_active

        if simulation_thread is None or not simulation_thread.is_alive():
            simulation_thread = threading.Thread(target=run_simulation, daemon=True)
            simulation_thread.start()
            logger.info("Simulation thread started")

    return 0


@app.callback(
    Output('refresh-interval', 'n_intervals', allow_duplicate=True),
    Input('stop-btn', 'n_clicks'),
    prevent_initial_call=True
)
def stop_simulation_callback(n_clicks):
    """Stop the simulation."""
    if n_clicks:
        global simulation_active
        simulation_active = False
        logger.info("Simulation stop requested")

    return 0


@app.callback(
    Output('log-text', 'children', allow_duplicate=True),
    Input('save-btn', 'n_clicks'),
    prevent_initial_call=True
)
def save_checkpoint_callback(n_clicks):
    """Save checkpoint."""
    if n_clicks:
        success, result = state.save_checkpoint()
        if success:
            return f"Checkpoint saved: {result}"
        else:
            return f"Save error: {result}"
    return "System ready."


@app.callback(
    Output('refresh-interval', 'n_intervals', allow_duplicate=True),
    [Input('step-interval', 'value'),
     Input('save-interval', 'value'),
     Input('auto-backup', 'value')],
    prevent_initial_call=True
)
def update_config(step_interval, save_interval, auto_backup):
    """Update configuration."""
    if step_interval:
        state.step_interval = max(0.01, float(step_interval))
    if save_interval:
        state.save_interval = max(10, int(save_interval))
    if auto_backup is not None:
        state.auto_backup = bool(auto_backup)

    logger.info(f"Config updated: interval={state.step_interval}s, save={state.save_interval}")
    return 0


def create_empty_figure(title: str):
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        height=250,
        annotations=[{
            "text": title,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 16, "color": "#6c757d"}
        }]
    )
    return fig


def main():
    """Run the dashboard application."""
    logger.info("Starting NeuralChild Dashboard...")
    app.run_server(debug=True, host='0.0.0.0', port=8050)


if __name__ == '__main__':
    main()
