from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def rank_antibiotics_for_bacteria(
    bacteria_name: str, artifact: Dict, top_n: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = artifact["best_model"]
    target_inverse = artifact["inverse_target_mapping"]
    bacteria_encoder = artifact["encoders"]["bacteria_encoder"]
    antibiotic_encoder = artifact["encoders"]["antibiotic_encoder"]

    if bacteria_name not in bacteria_encoder.classes_:
        raise ValueError(f"Bacteria '{bacteria_name}' not found in training classes.")

    bacteria_encoded = int(bacteria_encoder.transform([bacteria_name])[0])
    rows = []

    for antibiotic in antibiotic_encoder.classes_:
        antibiotic_encoded = int(antibiotic_encoder.transform([antibiotic])[0])
        pred_proba = model.predict_proba([[bacteria_encoded, antibiotic_encoded]])[0]
        pred_class = int(model.predict([[bacteria_encoded, antibiotic_encoded]])[0])

        resistant_prob = float(pred_proba[1]) if len(pred_proba) > 1 else 0.0
        rows.append(
            {
                "Antibiotic": antibiotic,
                "PredictedClassCode": pred_class,
                "PredictedClass": target_inverse[pred_class],
                "ResistanceProbability": resistant_prob,
                "SusceptibleProbability": float(pred_proba[0]),
                "IntermediateProbability": float(pred_proba[2])
                if len(pred_proba) > 2
                else 0.0,
            }
        )

    ranking = pd.DataFrame(rows).sort_values("ResistanceProbability", ascending=True)
    recommended = ranking.head(top_n).copy()
    avoid = ranking.sort_values("ResistanceProbability", ascending=False).head(top_n).copy()
    return recommended, avoid
