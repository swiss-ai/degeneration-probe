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

# Latest slider values, kept in sync by the .change() handlers. generate_stream
# reads this to attribute the live-temperature-at-token for the trace chart.
_current_values: dict[str, float] = {"temperature": 0.01, "top_p": 0.9, "steering_threshold": 0.8}


def _set_live_ws(ws: websocket.WebSocket | None) -> None:
    global _live_ws
    with _live_ws_lock:
        _live_ws = ws


def _push_param_update(name: str):
    """Return a Gradio .change() handler that forwards the new value to the worker."""
    def _send(value):
        _current_values[name] = float(value)
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


def fetch_model_label() -> str:
    """Ask the backend which model the worker is serving."""
    try:
        resp = requests.get(f"{API_BASE}/api/worker/info", timeout=10)
        if resp.ok:
            info = resp.json()
            probe = " + probe" if info.get("has_probe") else ""
            return (
                f'<span style="color:#8a8fa0;font-size:13px;">Model:</span> '
                f'<code style="background:#eef1f6;padding:2px 6px;border-radius:4px;'
                f'font-size:13px;">{info.get("model", "unknown")}{probe}</code>'
            )
        return f'<span style="color:#a05050;font-size:13px;">Worker info: {resp.text[:80]}</span>'
    except Exception as e:
        return f'<span style="color:#a05050;font-size:13px;">Worker unreachable: {e}</span>'


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
    scores: list[float] = []
    temperatures: list[float] = []
    ws = None

    # Snapshot the launch-time slider values so the mid-stream tracker starts
    # from the same numbers the user clicked Generate with.
    _current_values["temperature"] = float(temperature)
    _current_values["top_p"] = float(top_p)
    _current_values["steering_threshold"] = float(threshold)

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
                ), "", ""
                break

            if data["type"] == "done":
                log.info("Generation complete: %d tokens", len(scores))
                break

            if data["type"] == "tokens":
                for tok in data["tokens"]:
                    score = tok["probe_score"]
                    scores.append(score)
                    temperatures.append(_current_values["temperature"])
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
                bar_html = _build_score_bar(scores, _current_values["steering_threshold"])
                temp_html = _build_temp_chart(temperatures)
                yield tokens_html, bar_html, temp_html

    except Exception as e:
        log.exception("Connection error after %d tokens", len(scores))
        yield tokens_html + f'<br><span style="color:{COLORS["muted"]};">Connection error: {e}</span>', "", ""
    finally:
        _set_live_ws(None)
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


CHART_W = 600
CHART_H = 90


def _placeholder(msg: str) -> str:
    return (
        f'<div style="color:{COLORS["muted"]};font-style:italic;font-size:13px;'
        f'padding:22px;text-align:center;background:{COLORS["bar_bg"]};'
        f'border-radius:8px;">{msg}</div>'
    )


def _line_points(values: list[float], value_range: tuple[float, float]) -> str:
    """Map a series of values to SVG polyline points across CHART_W x CHART_H."""
    lo, hi = value_range
    span = max(hi - lo, 1e-9)
    n = len(values)
    if n == 1:
        y = CHART_H - ((values[0] - lo) / span) * CHART_H
        return f"0,{y:.2f} {CHART_W},{y:.2f}"
    parts = []
    for i, v in enumerate(values):
        x = i * CHART_W / (n - 1)
        y = CHART_H - (max(min(v, hi), lo) - lo) / span * CHART_H
        parts.append(f"{x:.2f},{y:.2f}")
    return " ".join(parts)


def _chart_frame(svg_body: str) -> str:
    return (
        f'<div style="background:{COLORS["bar_bg"]};border-radius:8px;padding:10px 12px;">'
        f'<svg viewBox="0 0 {CHART_W} {CHART_H}" preserveAspectRatio="none" '
        f'width="100%" height="{CHART_H}px" style="display:block;overflow:visible;">'
        f'{svg_body}'
        f'</svg>'
        f'</div>'
    )


def _gridline(y: float, stroke: str = "#dde2ec", dashed: bool = False) -> str:
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return f'<line x1="0" y1="{y:.2f}" x2="{CHART_W}" y2="{y:.2f}" stroke="{stroke}" stroke-width="1"{dash}/>'


def _axis_tick(y: float, label: str, anchor: str = "end") -> str:
    x = CHART_W - 4 if anchor == "end" else 4
    return (
        f'<text x="{x}" y="{y:.2f}" dy="3" text-anchor="{anchor}" '
        f'font-size="10" fill="{COLORS["muted"]}" font-family="system-ui,sans-serif">'
        f'{label}</text>'
    )


def _build_score_bar(scores: list[float], threshold: float) -> str:
    """Build a clean line chart of probe scores per token."""
    if not scores:
        return _placeholder("No tokens yet — scores will stream here")

    points = _line_points(scores, (0.0, 1.0))
    thresh_y = CHART_H - threshold * CHART_H
    current_color = _score_to_color(scores[-1])

    svg = "".join([
        # background gridlines
        _gridline(CHART_H * 0.5),
        # threshold
        f'<line x1="0" y1="{thresh_y:.2f}" x2="{CHART_W}" y2="{thresh_y:.2f}" '
        f'stroke="{COLORS["threshold_line"]}" stroke-width="1.2" stroke-dasharray="5 3"/>',
        # subtle fill below the line
        f'<polygon fill="{current_color}" fill-opacity="0.12" '
        f'points="0,{CHART_H} {points} {CHART_W},{CHART_H}"/>',
        # the line itself
        f'<polyline fill="none" stroke="{current_color}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{points}"/>',
        _axis_tick(10, "1.0"),
        _axis_tick(CHART_H - 2, "0.0"),
    ])

    avg_score = sum(scores) / len(scores)
    over = sum(1 for s in scores if s >= threshold)
    pct_over = 100.0 * over / len(scores)
    footer = (
        f'<div style="display:flex;justify-content:space-between;margin-top:8px;'
        f'font-size:12px;color:{COLORS["muted"]};flex-wrap:wrap;gap:12px;">'
        f'<span>Latest: <b style="color:{current_color}">{scores[-1]:.3f}</b></span>'
        f'<span>Mean: <b>{avg_score:.3f}</b></span>'
        f'<span>Above threshold ({threshold:.2f}): <b>{over}/{len(scores)}</b> '
        f'({pct_over:.0f}%)</span>'
        f'</div>'
    )
    return _chart_frame(svg) + footer


def _build_temp_chart(temperatures: list[float], max_temp: float = 2.0) -> str:
    """Build a clean line chart of the sampling temperature at each token."""
    if not temperatures:
        return _placeholder("Temperature trace appears once generation starts")

    points = _line_points(temperatures, (0.0, max_temp))
    norm = min(max(temperatures[-1], 0.0), max_temp) / max_temp
    r = int(74 + norm * 140)
    g = int(182 - norm * 95)
    b = int(144 - norm * 55)
    current_color = f"rgb({r},{g},{b})"

    svg = "".join([
        _gridline(CHART_H * 0.5),
        f'<polygon fill="{current_color}" fill-opacity="0.12" '
        f'points="0,{CHART_H} {points} {CHART_W},{CHART_H}"/>',
        f'<polyline fill="none" stroke="{current_color}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{points}"/>',
        _axis_tick(10, f"{max_temp:.1f}"),
        _axis_tick(CHART_H - 2, "0.0"),
    ])

    mean_t = sum(temperatures) / len(temperatures)
    footer = (
        f'<div style="display:flex;justify-content:space-between;margin-top:8px;'
        f'font-size:12px;color:{COLORS["muted"]};">'
        f'<span>Current: <b>{temperatures[-1]:.2f}</b></span>'
        f'<span>Mean: <b>{mean_t:.2f}</b></span>'
        f'<span>Tokens: <b>{len(temperatures)}</b></span>'
        f'</div>'
    )
    return _chart_frame(svg) + footer


def build_ui():
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(title="Degeneration Probe", css=CSS) as demo:
        gr.Markdown(
            '<h1 style="font-weight:600;color:#2d3142;margin-bottom:2px;">'
            'Degeneration Probe</h1>'
            '<p style="color:#8a8fa0;font-size:14px;margin-top:0;">'
            'Real-time degeneration detection and model steering</p>'
        )
        model_label = gr.HTML(value="")

        demo.load(ensure_session).then(fetch_model_label, outputs=[model_label])

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

                gr.Markdown(
                    '<div style="font-weight:600;font-size:14px;color:#2d3142;'
                    'margin-top:18px;margin-bottom:2px;">Degeneration Score</div>'
                    '<div style="font-size:12px;color:#8a8fa0;margin-bottom:8px;">'
                    'Per-token probe output (0 = natural, 1 = degenerate). '
                    'The dashed line marks the steering threshold.</div>'
                )
                score_bar = gr.HTML(
                    value=_build_score_bar([], 0.8),
                    elem_id="score-panel",
                )

                with gr.Accordion("Temperature trace", open=False):
                    gr.Markdown(
                        '<div style="font-size:12px;color:#8a8fa0;margin-bottom:8px;">'
                        'Sampling temperature at each generated token. '
                        'Moves when you slide the Temperature control mid-stream.</div>'
                    )
                    temp_chart = gr.HTML(
                        value=_build_temp_chart([]),
                        elem_id="temp-panel",
                    )

        gen_event = generate_btn.click(
            generate_stream,
            inputs=[
                prompt_input, temperature, top_p, max_tokens,
                steering_enabled, strategy, threshold, boost_temp,
            ],
            outputs=[output_html, score_bar, temp_chart],
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
