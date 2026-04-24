from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image

from src.pipeline import predict
from src.viz import build_prediction_map


def _wrap_map_html(raw_html: str) -> str:
    escaped = html.escape(raw_html, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%; height:520px; border:0;"></iframe>'
    )


def _format_scripts(result: dict) -> str:
    if not result.get("ocr_enabled"):
        return "(OCR disabled)"
    scripts = result.get("detected_scripts")
    if not scripts:
        return "(no text detected)"
    ordered = sorted(scripts.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{k} ({v:.2f})" for k, v in ordered)


def run(
    image: Image.Image | None,
    top_k: int,
    top_cells: int,
    enable_ocr: bool,
    enable_color: bool,
    beta: float,
    gamma: float,
    classifier_path: str,
    prototypes_path: str,
    model_name: str,
):
    if image is None:
        return (
            "Please upload an image.",
            "(no image)",
            "<div>No image.</div>",
            [],
        )
    result = predict(
        image,
        classifier_path=classifier_path,
        prototypes_path=prototypes_path,
        top_cells=int(top_cells),
        top_k=int(top_k),
        model_name=model_name,
        enable_ocr=bool(enable_ocr),
        enable_color=bool(enable_color),
        alpha=1.0,
        beta=float(beta),
        gamma=float(gamma),
    )
    lat, lon = result["predicted"]
    conf = result["confidence"]
    summary = f"Predicted: ({lat:.4f}, {lon:.4f})   confidence={conf:.3f}"
    scripts_text = _format_scripts(result)
    map_html = build_prediction_map(
        predicted_lat=lat,
        predicted_lon=lon,
        candidates=result["top_k"],
        cell_centroids=result["cell_probs"],
    )
    rows = []
    for c in result["top_k"]:
        sb = "—" if c.get("script_bonus") is None else f"{c['script_bonus']:+.1f}"
        cs = "—" if c.get("color_sim") is None else f"{c['color_sim']:.3f}"
        rows.append(
            [
                c["rank"],
                c.get("country") or "?",
                f"{c['lat']:.4f}",
                f"{c['lon']:.4f}",
                f"{c['confidence']:.3f}",
                f"{c['similarity']:+.3f}",
                sb,
                cs,
                f"{c['final_score']:+.3f}",
                c["cell_id"],
            ]
        )
    return summary, scripts_text, _wrap_map_html(map_html), rows


def build_ui(
    default_classifier: str,
    default_prototypes: str,
    default_model: str,
    default_beta: float,
    default_gamma: float,
) -> gr.Blocks:
    with gr.Blocks(title="PIDGEOT — image geolocalization") as demo:
        gr.Markdown(
            "# PIDGEOT\n"
            "Upload a street-level photo to predict where on Earth it was taken. "
            "Pipeline: StreetCLIP + semantic geocells + haversine-smoothed classifier "
            "+ prototype refinement + optional OCR-script and color-histogram re-ranking."
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="pil", label="Query image")
                top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K candidates")
                top_cells = gr.Slider(1, 20, value=5, step=1, label="Top-cells to refine")
                enable_ocr = gr.Checkbox(value=True, label="Enable OCR re-ranking")
                beta = gr.Slider(
                    0.0, 2.0, value=float(default_beta), step=0.05,
                    label="β (OCR weight)",
                )
                enable_color = gr.Checkbox(value=True, label="Enable color-histogram re-ranking")
                gamma = gr.Slider(
                    0.0, 2.0, value=float(default_gamma), step=0.05,
                    label="γ (color-histogram weight)",
                )
                classifier_path = gr.Textbox(value=default_classifier, label="Classifier path")
                prototypes_path = gr.Textbox(value=default_prototypes, label="Prototypes path")
                model_name = gr.Textbox(value=default_model, label="StreetCLIP model")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Textbox(label="Prediction", interactive=False)
                scripts_out = gr.Textbox(label="Detected scripts", interactive=False)
                map_html = gr.HTML(label="Map")
                table = gr.Dataframe(
                    headers=[
                        "rank",
                        "country",
                        "lat",
                        "lon",
                        "confidence",
                        "sim",
                        "script_bonus",
                        "color_sim",
                        "final_score",
                        "cell_id",
                    ],
                    label="Top-K candidates",
                    interactive=False,
                )
        btn.click(
            run,
            inputs=[
                img_in,
                top_k,
                top_cells,
                enable_ocr,
                enable_color,
                beta,
                gamma,
                classifier_path,
                prototypes_path,
                model_name,
            ],
            outputs=[summary, scripts_out, map_html, table],
        )
    return demo


def _load_defaults(config_path: str) -> tuple[float, float]:
    p = Path(config_path)
    if not p.exists():
        return 0.5, 0.3
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        rr = raw.get("rerank", {}) or {}
        return float(rr.get("beta", 0.5)), float(rr.get("gamma", 0.3))
    except Exception:
        return 0.5, 0.3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default=os.environ.get("PIDGEOT_CLASSIFIER", "data/classifier.pt"))
    parser.add_argument("--prototypes", type=str, default=os.environ.get("PIDGEOT_PROTOTYPES", "data/prototypes.npz"))
    parser.add_argument("--model", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not Path(args.classifier).exists():
        print(f"[app] warning: classifier not found at {args.classifier}")
    if not Path(args.prototypes).exists():
        print(f"[app] warning: prototypes not found at {args.prototypes}")

    default_beta, default_gamma = _load_defaults(args.config)
    demo = build_ui(args.classifier, args.prototypes, args.model, default_beta, default_gamma)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
