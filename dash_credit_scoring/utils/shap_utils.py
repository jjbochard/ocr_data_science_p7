import numpy as np
import shap
from scipy.special import expit


def transform_shap_to_proba(shap_values, pred_proba, i):
    """
    Transform SHAP values from log-odds to approximate probability scale.
    """
    base_value_logodds = shap_values.base_values[i]
    base_proba = expit(base_value_logodds)

    # Calculate explain distance
    delta_model = pred_proba[i] - base_proba
    delta_shap = np.sum(shap_values.values[i])

    # Ratio
    scale = delta_shap / delta_model if delta_model != 0 else 1

    # Rescale contributions
    values_scaled = shap_values.values[i] / scale

    return shap.Explanation(
        values=values_scaled,
        base_values=base_proba,
        data=shap_values.data[i],
        feature_names=shap_values.feature_names,
    )
