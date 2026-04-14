"""Gradio interface for the degeneration probe visualization."""

from __future__ import annotations

import json
import logging
import threading

import gradio as gr
import requests
import websocket  # from websocket-client

log = logging.getLogger("ui")

API_BASE = "http://localhost:8000"

# Module-level handle on the currently open generation WebSocket so the slider
# .change() handlers (which run outside generate_stream's scope) can push live
# parameter updates through it.
_live_ws: websocket.WebSocket | None = None
_live_ws_lock = threading.Lock()


def _set_live_ws(ws: websocket.WebSocket | None) -> None:
    global _live_ws
    with _live_ws_lock:
        _live_ws = ws


def _push_param_update(name: str):
    """Return a Gradio .change() handler that forwards the new value to the worker."""
    def _send(value):
        with _live_ws_lock:
            ws = _live_ws
        if ws is None:
            log.info("live %s=%s: no active generation, skipping", name, value)
            return
        try:
            ws.send(json.dumps({"action": "update_params", "params": {name: value}}))
            log.info("live %s=%s: pushed to worker", name, value)
        except Exception as e:
            log.warning("live %s=%s: send failed: %s", name, value, e)
    return _send

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


def ensure_session():
    """Ensure the backend has a session pointing at the local tunnel.

    The worker is always reached through the SSH tunnel on localhost:9000
    that cluster/start.sh sets up, so the UI no longer exposes host/port.
    """
    try:
        requests.post(
            f"{API_BASE}/api/sessions",
            json={"worker_host": "localhost", "worker_port": 9000},
            timeout=5,
        )
    except Exception as e:
        log.warning("Could not create backend session: %s", e)


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
        log.info("Connecting to backend WebSocket at %s", ws_url)
        ws = websocket.create_connection(ws_url, timeout=5)
        ws.settimeout(300)  # 5 min recv timeout (token gen can be slow on CPU)
        _set_live_ws(ws)
        ws.send(json.dumps(request))
        log.info("Request sent, waiting for tokens...")

        while True:
            raw = ws.recv()
            data = json.loads(raw)

            if data["type"] == "error":
                log.error("Error from backend: %s", data["message"])
                yield (
                    tokens_html
                    + f'<span style="color:{_score_to_color(1.0)};font-weight:500;">'
                    + f' Error: {data["message"]}</span>'
                ), ""
                break

            if data["type"] == "done":
                log.info("Generation complete: %d tokens", len(scores))
                break

            if data["type"] == "tokens":
                for tok in data["tokens"]:
                    score = tok["probe_score"]
                    scores.append(score)
                    steered = tok["was_steered"]

                    color = _score_to_color(score)
                    style = f"color:{color};"
                    if steered:
                        sc = COLORS["steered"]
                        style += (
                            f"text-decoration:underline;"
                            f"text-decoration-color:rgb({sc[0]},{sc[1]},{sc[2]});"
                            f"text-underline-offset:3px;"
                        )

                    token_text = tok["token_text"].replace("<", "&lt;").replace(">", "&gt;")
                    tokens_html += (
                        f'<span style="{style}" title="score={score:.3f}">'
                        f'{token_text}</span>'
                    )

                # One yield per batch amortises the Gradio SSE / browser DOM cost.
                bar_html = _build_score_bar(scores, threshold)
                yield tokens_html, bar_html

    except Exception as e:
        log.exception("Connection error after %d tokens", len(scores))
        yield tokens_html + f'<br><span style="color:{COLORS["muted"]};">Connection error: {e}</span>', ""
    finally:
        _set_live_ws(None)
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

        demo.load(ensure_session)

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

        # Live-update sliders: push the new value to the worker via the open
        # generation WebSocket. No-op when nothing is running.
        temperature.change(_push_param_update("temperature"), inputs=[temperature], show_progress="hidden")
        top_p.change(_push_param_update("top_p"), inputs=[top_p], show_progress="hidden")
        threshold.change(_push_param_update("steering_threshold"), inputs=[threshold], show_progress="hidden")

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
