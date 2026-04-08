"""Gradio interface for the degeneration probe visualization."""

from __future__ import annotations

import json

import gradio as gr
import requests
import websocket  # from websocket-client, bundled with gradio

API_BASE = "http://localhost:8000"


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
    ws_url = f"ws://localhost:8000/api/generate"

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

    try:
        ws = websocket.create_connection(ws_url)
        ws.send(json.dumps(request))

        while True:
            raw = ws.recv()
            data = json.loads(raw)

            if data["type"] == "error":
                yield tokens_html + f"<br><b>Error:</b> {data['message']}", ""
                break

            if data["type"] == "done":
                break

            if data["type"] == "token":
                score = data["probe_score"]
                scores.append(score)
                steered = data["was_steered"]

                # Color: green (0) -> yellow (0.5) -> red (1.0)
                if score < 0.5:
                    r = int(255 * score * 2)
                    g = 200
                else:
                    r = 255
                    g = int(200 * (1 - (score - 0.5) * 2))
                color = f"rgb({r},{g},0)"

                style = f"color:{color};"
                if steered:
                    style += "text-decoration:underline;"

                token_text = data["token_text"].replace("<", "&lt;").replace(">", "&gt;")
                tokens_html += f'<span style="{style}" title="score={score:.3f}">{token_text}</span>'

                # Build sparkline as simple bar chart
                bar_html = _build_score_bar(scores, threshold)

                yield tokens_html, bar_html

        ws.close()

    except Exception as e:
        yield tokens_html + f"<br><b>Connection error:</b> {e}", ""


def _build_score_bar(scores: list[float], threshold: float) -> str:
    """Build an HTML sparkline bar chart of probe scores."""
    if not scores:
        return ""
    bar_width = max(2, min(6, 600 // len(scores)))
    bars = []
    for s in scores:
        height = max(2, int(s * 60))
        if s < 0.5:
            r = int(255 * s * 2)
            g = 200
        else:
            r = 255
            g = int(200 * (1 - (s - 0.5) * 2))
        color = f"rgb({r},{g},0)"
        bars.append(
            f'<div style="display:inline-block;width:{bar_width}px;height:{height}px;'
            f'background:{color};vertical-align:bottom;"></div>'
        )
    threshold_y = int(threshold * 60)
    return (
        f'<div style="position:relative;height:65px;overflow-x:auto;white-space:nowrap;'
        f'border-bottom:1px solid #ccc;padding-top:5px;">'
        f'<div style="position:absolute;top:{65-threshold_y}px;left:0;right:0;'
        f'border-top:2px dashed rgba(255,0,0,0.5);"></div>'
        f'{"".join(bars)}'
        f'</div>'
        f'<div style="font-size:12px;color:#666;">Current: {scores[-1]:.3f} | Threshold: {threshold:.2f}</div>'
    )


def build_ui():
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(title="Degeneration Probe", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Degeneration Probe — Live Steering")

        # Connection bar
        with gr.Row():
            host_input = gr.Textbox(value="localhost", label="Worker Host", scale=2)
            port_input = gr.Number(value=9000, label="Port", precision=0, scale=1)
            connect_btn = gr.Button("Connect", scale=1)
            disconnect_btn = gr.Button("Disconnect", scale=1)
            status_display = gr.Textbox(label="Status", interactive=False, scale=2)

        connect_btn.click(
            connect_worker, inputs=[host_input, port_input], outputs=[status_display]
        )
        disconnect_btn.click(disconnect_worker, outputs=[status_display])
        demo.load(get_status, outputs=[status_display])

        # Prompt input
        with gr.Row():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter a prompt or select from dataset samples...",
                lines=3,
                scale=4,
            )
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            stop_btn = gr.Button("Stop")

        # Controls + Output
        with gr.Row():
            # Left: controls
            with gr.Column(scale=1):
                gr.Markdown("### Generation Parameters")
                temperature = gr.Slider(0.0, 2.0, value=0.01, step=0.01, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                max_tokens = gr.Slider(64, 4096, value=512, step=64, label="Max Tokens")

                gr.Markdown("### Steering")
                steering_enabled = gr.Checkbox(label="Enable Steering", value=False)
                strategy = gr.Dropdown(
                    choices=["temperature_boost"],
                    value="temperature_boost",
                    label="Strategy",
                )
                threshold = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Threshold")
                boost_temp = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="Boost Temperature")

            # Right: streaming output
            with gr.Column(scale=3):
                output_html = gr.HTML(label="Generated Text")
                score_bar = gr.HTML(label="Probe Score")

        generate_btn.click(
            generate_stream,
            inputs=[
                prompt_input, temperature, top_p, max_tokens,
                steering_enabled, strategy, threshold, boost_temp,
            ],
            outputs=[output_html, score_bar],
        )

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
