from __future__ import annotations

from typing import Iterable

import folium


def build_prediction_map(
    predicted_lat: float,
    predicted_lon: float,
    candidates: Iterable[dict],
    cell_centroids: Iterable[dict] | None = None,
    zoom_start: int = 3,
) -> str:
    m = folium.Map(location=[predicted_lat, predicted_lon], zoom_start=zoom_start, control_scale=True)

    folium.Marker(
        location=[predicted_lat, predicted_lon],
        tooltip="Prediction",
        popup=f"Prediction<br>({predicted_lat:.4f}, {predicted_lon:.4f})",
        icon=folium.Icon(color="red", icon="star"),
    ).add_to(m)

    for c in candidates:
        lat = c["lat"]
        lon = c["lon"]
        rank = c.get("rank", "?")
        conf = c.get("confidence", 0.0)
        cid = c.get("cell_id", "?")
        if lat == predicted_lat and lon == predicted_lon and rank == 1:
            continue
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="#1f78b4",
            fill=True,
            fill_color="#1f78b4",
            fill_opacity=0.8,
            popup=(
                f"#{rank}<br>({lat:.4f}, {lon:.4f})<br>"
                f"confidence {conf:.3f}<br>cell {cid}"
            ),
            tooltip=f"#{rank} conf={conf:.2f}",
        ).add_to(m)

    if cell_centroids:
        for cp in cell_centroids:
            folium.CircleMarker(
                location=[cp["centroid_lat"], cp["centroid_lon"]],
                radius=max(3, float(cp.get("prob", 0.0)) * 40),
                color="#888",
                weight=1,
                fill=True,
                fill_opacity=0.15,
                popup=f"cell {cp['cell_id']} prob={cp.get('prob', 0.0):.3f}",
            ).add_to(m)

    m.get_root().width = "100%"
    m.get_root().height = "500px"
    return m.get_root().render()
