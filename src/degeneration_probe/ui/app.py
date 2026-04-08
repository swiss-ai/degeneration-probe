"""Gradio interface for the degeneration probe visualization."""

from __future__ import annotations

import json

import gradio as gr
import requests
import websocket  # from websocket-client

API_BASE = "http://localhost:8000"

# Muted, modern palette
COLORS = {
    "safe": (74, 182, 144),       # soft teal-green
    "mid": (234, 179, 70),        # warm amber
    "danger": (214, 87, 89),      # muted coral-red
    "steered": (130, 120, 210),   # soft purple underline
    "bg": "#f8f9fb",
    "card": "#ffffff",
    "border": "#e2e6ea",
    "text": "#2d3142",
    "muted": "#8a8fa0",
    "bar_bg": "#f0f1f4",
    "threshold_line": "rgba(214,87,89,0.45)",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

.gradio-container {
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace !important;
    max-width: 1120px !important;
    background: %(bg)s !important;
}
.gr-button-primary {
    background: #4a6cf7 !important;
    border: none !important;
    border-radius: 8px !important;
}
.gr-button-secondary, .gr-button {
    border-radius: 8px !important;
}
#output-panel {
    background: %(card)s;
    border: 1px solid %(border)s;
    border-radius: 12px;
    padding: 20px 24px;
    min-height: 200px;
    font-size: 15px;
    line-height: 1.7;
    color: %(text)s;
}
#score-panel {
    background: %(card)s;
    border: 1px solid %(border)s;
    border-radius: 12px;
    padding: 14px 20px;
    margin-top: 8px;
}
""" % COLORS


def _score_to_color(score: float) -> str:
    """Map probe score 0-1 to a smooth gradient through the palette."""
    if score < 0.5:
        t = score * 2
        r = int(COLORS["safe"][0] + (COLORS["mid"][0] - COLORS["safe"][0]) * t)
        g = int(COLORS["safe"][1] + (COLORS["mid"][1] - COLORS["safe"][1]) * t)
        b = int(COLORS["safe"][2] + (COLORS["mid"][2] - COLORS["safe"][2]) * t)
    else:
        t = (score - 0.5) * 2
        r = int(COLORS["mid"][0] + (COLORS["danger"][0] - COLORS["mid"][0]) * t)
        g = int(COLORS["mid"][1] + (COLORS["danger"][1] - COLORS["mid"][1]) * t)
        b = int(COLORS["mid"][2] + (COLORS["danger"][2] - COLORS["mid"][2]) * t)
    return f"rgb({r},{g},{b})"


def connect_worker(host: str, port: int):
    """Connect to an inference worker."""
    try:
        resp = requests.post(
            f"{API_BASE}/api/sessions",
            json={"worker_host": host, "worker_port": int(port)},
            timeout=5,
        )
        if resp.ok:
            return f"Connected to {host}:{int(port)}"
        return f"Failed: {resp.text}"
    except Exception as e:
        return f"Connection error: {e}"


def disconnect_worker():
    """Disconnect from the current worker."""
    try:
        resp = requests.delete(f"{API_BASE}/api/sessions/current", timeout=5)
        return "Disconnected" if resp.ok else f"Failed: {resp.text}"
    except Exception as e:
        return f"Error: {e}"


def get_status():
    """Check current connection status."""
    try:
        resp = requests.get(f"{API_BASE}/api/sessions/current", timeout=2)
        if resp.ok:
            s = resp.json()
            return f"Connected to {s['worker_host']}:{s['worker_port']}"
        return "Disconnected"
    except Exception:
        return "Backend unavailable"


def generate_stream(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    steering_enabled: bool,
    strategy: str,
    threshold: float,
    boost_temp: float,
):
    """Stream generation from the worker via the backend WebSocket."""
    ws_url = "ws://localhost:8000/api/generate"

    request = {
        "action": "generate",
        "prompt": prompt,
        "params": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": int(max_tokens),
        },
        "steering": {
            "enabled": steering_enabled,
            "strategy": strategy,
            "threshold": threshold,
            "boost_temperature": boost_temp,
        },
    }

    tokens_html = ""
    scores = []
    ws = None

    try:
        ws = websocket.create_connection(ws_url, timeout=5)
        ws.send(json.dumps(request))

        while True:
            raw = ws.recv()
            data = json.loads(raw)

            if data["type"] == "error":
                yield (
                    tokens_html
                    + f'<span style="color:{_score_to_color(1.0)};font-weight:500;">'
                    + f' Error: {data["message"]}</span>'
                ), ""
                break

            if data["type"] == "done":
                break

            if data["type"] == "token":
                score = data["probe_score"]
                scores.append(score)
                steered = data["was_steered"]

                color = _score_to_color(score)
                style = f"color:{color};"
                if steered:
                    sc = COLORS["steered"]
                    style += (
                        f"text-decoration:underline;"
                        f"text-decoration-color:rgb({sc[0]},{sc[1]},{sc[2]});"
                        f"text-underline-offset:3px;"
                    )

                token_text = data["token_text"].replace("<", "&lt;").replace(">", "&gt;")
                tokens_html += (
                    f'<span style="{style}" title="score={score:.3f}">'
                    f'{token_text}</span>'
                )

                bar_html = _build_score_bar(scores, threshold)
                yield tokens_html, bar_html

    except Exception as e:
        yield tokens_html + f'<br><span style="color:{COLORS["muted"]};">Connection error: {e}</span>', ""
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


def _build_score_bar(scores: list[float], threshold: float) -> str:
    """Build an HTML sparkline bar chart of probe scores."""
    if not scores:
        return ""
    bar_width = max(2, min(5, 580 // len(scores)))
    bars = []
    for s in scores:
        height = max(2, int(s * 52))
        color = _score_to_color(s)
        bars.append(
            f'<div style="display:inline-block;width:{bar_width}px;height:{height}px;'
            f'background:{color};vertical-align:bottom;border-radius:1px;'
            f'margin-right:1px;"></div>'
        )
    threshold_y = int(threshold * 52)
    return (
        f'<div style="position:relative;height:58px;overflow-x:auto;white-space:nowrap;'
        f'background:{COLORS["bar_bg"]};border-radius:8px;padding:4px 8px;">'
        f'<div style="position:absolute;top:{58-threshold_y}px;left:0;right:0;'
        f'border-top:1.5px dashed {COLORS["threshold_line"]};"></div>'
        f'{"".join(bars)}'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:6px;'
        f'font-size:12px;color:{COLORS["muted"]};">'
        f'<span>Current score: <b style="color:{_score_to_color(scores[-1])}">{scores[-1]:.3f}</b></span>'
        f'<span>Threshold: {threshold:.2f}</span>'
        f'</div>'
    )


def build_ui():
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(title="Degeneration Probe", css=CSS) as demo:
        gr.Markdown(
            '<h1 style="font-weight:600;color:#2d3142;margin-bottom:2px;">'
            'Degeneration Probe</h1>'
            '<p style="color:#8a8fa0;font-size:14px;margin-top:0;">'
            'Real-time degeneration detection and model steering</p>'
        )

        # Connection bar
        with gr.Row():
            host_input = gr.Textbox(value="localhost", label="Worker Host", scale=2)
            port_input = gr.Number(value=9000, label="Port", precision=0, scale=1)
            connect_btn = gr.Button("Connect", variant="secondary", scale=1)
            disconnect_btn = gr.Button("Disconnect", variant="secondary", scale=1)
            status_display = gr.Textbox(label="Status", interactive=False, scale=2)

        connect_btn.click(
            connect_worker, inputs=[host_input, port_input], outputs=[status_display]
        )
        disconnect_btn.click(disconnect_worker, outputs=[status_display])
        demo.load(get_status, outputs=[status_display])

        gr.Markdown("---")

        # Prompt input
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Type a prompt to generate from...",
            lines=2,
        )
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary", scale=3)
            stop_btn = gr.Button("Stop", variant="stop", scale=1)

        # Controls + Output
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown(
                    '<p style="font-weight:600;font-size:14px;color:#2d3142;margin-bottom:4px;">'
                    'Generation</p>'
                )
                temperature = gr.Slider(
                    0.0, 2.0, value=0.01, step=0.01, label="Temperature",
                    info="Low = deterministic, high = creative",
                )
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                max_tokens = gr.Slider(64, 4096, value=512, step=64, label="Max tokens")

                gr.Markdown(
                    '<p style="font-weight:600;font-size:14px;color:#2d3142;'
                    'margin-bottom:4px;margin-top:12px;">Steering</p>'
                )
                steering_enabled = gr.Checkbox(label="Enable steering", value=False)
                strategy = gr.Dropdown(
                    choices=["temperature_boost"],
                    value="temperature_boost",
                    label="Strategy",
                )
                threshold = gr.Slider(
                    0.0, 1.0, value=0.8, step=0.05, label="Threshold",
                    info="Probe score above this triggers intervention",
                )
                boost_temp = gr.Slider(
                    1.0, 5.0, value=1.5, step=0.1, label="Boost temperature",
                )

            with gr.Column(scale=3):
                output_html = gr.HTML(
                    value='<div style="color:#8a8fa0;font-style:italic;">Output will appear here...</div>',
                    elem_id="output-panel",
                    label="Generated Text",
                )
                score_bar = gr.HTML(
                    elem_id="score-panel",
                    label="Probe Score",
                )

        gen_event = generate_btn.click(
            generate_stream,
            inputs=[
                prompt_input, temperature, top_p, max_tokens,
                steering_enabled, strategy, threshold, boost_temp,
            ],
            outputs=[output_html, score_bar],
        )
        stop_btn.click(None, cancels=[gen_event])

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
