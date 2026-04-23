from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

import gradio as gr
from PIL import Image

from src.pipeline import predict
from src.viz import build_prediction_map


def _wrap_map_html(raw_html: str) -> str:
    escaped = html.escape(raw_html, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%; height:520px; border:0;"></iframe>'
    )


def run(
    image: Image.Image | None,
    top_k: int,
    top_cells: int,
    classifier_path: str,
    prototypes_path: str,
    model_name: str,
):
    if image is None:
        return "Please upload an image.", "<div>No image.</div>", []
    result = predict(
        image,
        classifier_path=classifier_path,
        prototypes_path=prototypes_path,
        top_cells=int(top_cells),
        top_k=int(top_k),
        model_name=model_name,
    )
    lat, lon = result["predicted"]
    conf = result["confidence"]
    summary = f"Predicted: ({lat:.4f}, {lon:.4f})   confidence={conf:.3f}"
    map_html = build_prediction_map(
        predicted_lat=lat,
        predicted_lon=lon,
        candidates=result["top_k"],
        cell_centroids=result["cell_probs"],
    )
    rows = [
        [c["rank"], f"{c['lat']:.4f}", f"{c['lon']:.4f}", f"{c['confidence']:.3f}", c["cell_id"]]
        for c in result["top_k"]
    ]
    return summary, _wrap_map_html(map_html), rows


def build_ui(default_classifier: str, default_prototypes: str, default_model: str) -> gr.Blocks:
    with gr.Blocks(title="PIDGEOT — image geolocalization") as demo:
        gr.Markdown(
            "# PIDGEOT\n"
            "Upload a street-level photo to predict where on Earth it was taken "
            "(StreetCLIP + semantic geocells + haversine-smoothed classifier + prototype refinement)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="pil", label="Query image")
                top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K candidates")
                top_cells = gr.Slider(1, 20, value=5, step=1, label="Top-cells to refine")
                classifier_path = gr.Textbox(value=default_classifier, label="Classifier path")
                prototypes_path = gr.Textbox(value=default_prototypes, label="Prototypes path")
                model_name = gr.Textbox(value=default_model, label="StreetCLIP model")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Textbox(label="Prediction", interactive=False)
                map_html = gr.HTML(label="Map")
                table = gr.Dataframe(
                    headers=["rank", "lat", "lon", "confidence", "cell_id"],
                    label="Top-K candidates",
                    interactive=False,
                )
        btn.click(
            run,
            inputs=[img_in, top_k, top_cells, classifier_path, prototypes_path, model_name],
            outputs=[summary, map_html, table],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default=os.environ.get("PIDGEOT_CLASSIFIER", "data/classifier.pt"))
    parser.add_argument("--prototypes", type=str, default=os.environ.get("PIDGEOT_PROTOTYPES", "data/prototypes.npz"))
    parser.add_argument("--model", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not Path(args.classifier).exists():
        print(f"[app] warning: classifier not found at {args.classifier}")
    if not Path(args.prototypes).exists():
        print(f"[app] warning: prototypes not found at {args.prototypes}")

    demo = build_ui(args.classifier, args.prototypes, args.model)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
