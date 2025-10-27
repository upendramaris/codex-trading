"""
Feature engineering helpers for advanced forecasting models.

This module currently focuses on market regime detection using Hidden Markov Models (HMMs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class RegimeDetectionResult:
    """Container for regime detection outputs."""

    states: pd.Series
    probabilities: pd.DataFrame
    model: GaussianHMM

    def dominant_state(self) -> int:
        """Return the state with the highest average posterior probability."""
        avg_probs = self.probabilities.mean(axis=0)
        return int(avg_probs.idxmax())


class RegimeDetector:
    """
    Hidden Markov Model based regime classifier.

    The detector fits a Gaussian HMM to asset returns and produces a sequence of hidden
    states together with smoothed posterior probabilities. States are ordered by their
    expected return (ascending) to encourage consistency across fits.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.state_order_: Optional[np.ndarray] = None

    def fit(self, returns: pd.Series) -> RegimeDetectionResult:
        """Fit the HMM to a returns series."""
        cleaned_returns = self._prepare_returns(returns)
        if len(cleaned_returns) < self.n_components * 5:
            raise ValueError("Insufficient observations to fit the HMM.")

        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        model.fit(cleaned_returns.values.reshape(-1, 1))

        ordered_model, order = self._order_states(model)
        self.model = ordered_model
        self.state_order_ = order

        states = ordered_model.predict(cleaned_returns.values.reshape(-1, 1))
        probabilities = ordered_model.predict_proba(cleaned_returns.values.reshape(-1, 1))

        state_series = pd.Series(states, index=cleaned_returns.index, name="regime")
        prob_df = pd.DataFrame(
            probabilities,
            index=cleaned_returns.index,
            columns=[f"regime_{i}" for i in range(ordered_model.n_components)],
        )
        return RegimeDetectionResult(state_series, prob_df, ordered_model)

    def transform(self, returns: pd.Series) -> RegimeDetectionResult:
        """Apply a fitted model to new returns."""
        if self.model is None:
            raise RuntimeError("RegimeDetector must be fitted before calling transform.")

        cleaned_returns = self._prepare_returns(returns)
        states = self.model.predict(cleaned_returns.values.reshape(-1, 1))
        probabilities = self.model.predict_proba(cleaned_returns.values.reshape(-1, 1))

        state_series = pd.Series(states, index=cleaned_returns.index, name="regime")
        prob_df = pd.DataFrame(
            probabilities,
            index=cleaned_returns.index,
            columns=[f"regime_{i}" for i in range(self.model.n_components)],
        )
        return RegimeDetectionResult(state_series, prob_df, self.model)

    def fit_transform(self, returns: pd.Series) -> RegimeDetectionResult:
        """Fit the model and return the detection result in one step."""
        return self.fit(returns)

    # ------------------------------------------------------------------
    def _prepare_returns(self, returns: pd.Series) -> pd.Series:
        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] != 1:
                raise ValueError("Returns DataFrame must contain exactly one column.")
            returns = returns.iloc[:, 0]
        returns = returns.dropna().astype(float)
        if returns.empty:
            raise ValueError("Returns series is empty after cleaning.")
        return returns

    def _order_states(self, model: GaussianHMM) -> tuple[GaussianHMM, np.ndarray]:
        """Reorder HMM states by ascending mean return to stabilise regime labels."""
        means = model.means_.flatten()
        order = np.argsort(means)

        model.startprob_ = model.startprob_[order]
        model.transmat_ = model.transmat_[np.ix_(order, order)]
        model.means_ = model.means_[order]

        if model.covariance_type in {"spherical", "diag"}:
            model.covars_ = model.covars_[order]
        else:
            model.covars_ = model.covars_[order]

        return model, order
