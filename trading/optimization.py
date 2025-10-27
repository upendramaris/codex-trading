"""
Portfolio optimisation utilities blending quantum-inspired search with classical models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

from ml.forecasting import RegimeDetectionResult


class CovarianceEstimator:
    """Exponentially weighted covariance estimator suitable for real-time updates."""

    def __init__(self, window: int = 252, decay: float = 0.97, min_periods: int = 60):
        self.window = window
        self.decay = decay
        self.min_periods = min_periods

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        recent = returns.tail(self.window).dropna()
        if len(recent) < self.min_periods:
            raise ValueError("Not enough observations for covariance estimation.")
        weights = np.power(self.decay, np.arange(len(recent))[::-1])
        weights /= weights.sum()

        mean = np.average(recent.values, axis=0, weights=weights)
        demeaned = recent.values - mean
        weighted = demeaned * np.sqrt(weights)[:, None]
        cov = weighted.T @ weighted
        cov = cov / (1 - self.decay)
        return pd.DataFrame(cov, index=recent.columns, columns=recent.columns)


class FactorModelPCA:
    """Simple PCA-based factor model for risk decomposition."""

    def __init__(self, n_factors: int = 3):
        self.n_factors = n_factors
        self.pca: Optional[PCA] = None
        self.explained_variance_: Optional[np.ndarray] = None

    def fit(self, returns: pd.DataFrame) -> None:
        n_components = min(self.n_factors, returns.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(returns.values)
        self.explained_variance_ = self.pca.explained_variance_ratio_

    def transform(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.pca is None:
            raise RuntimeError("FactorModelPCA must be fitted before calling transform.")
        factors = self.pca.transform(returns.values)
        loadings = self.pca.components_.T
        factor_df = pd.DataFrame(factors, index=returns.index, columns=[f"factor_{i}" for i in range(factors.shape[1])])
        loading_df = pd.DataFrame(loadings, index=returns.columns, columns=factor_df.columns)
        return factor_df, loading_df


class HRPAllocator:
    """Hierarchical Risk Parity allocator using correlation clustering."""

    @staticmethod
    def allocate(cov: pd.DataFrame) -> pd.Series:
        corr = HRPAllocator._cov_to_corr(cov)
        dist = np.sqrt((1 - corr).clip(0, 2))
        dist = squareform(dist.values, checks=False)
        linkage_matrix = linkage(dist, method="ward")

        sorted_indices = HRPAllocator._get_quasi_diag(linkage_matrix, cov.columns)
        weights = HRPAllocator._get_recursive_bisection(cov, sorted_indices)
        return weights.sort_index()

    @staticmethod
    def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
        std = np.sqrt(np.diag(cov))
        corr = cov / std[:, None] / std[None, :]
        return pd.DataFrame(corr, index=cov.index, columns=cov.columns)

    @staticmethod
    def _get_quasi_diag(linkage_matrix: np.ndarray, assets: Iterable[str]) -> list:
        assets = list(assets)
        sort_order = [len(linkage_matrix) * 2]

        def recurse(node_index):
            if node_index < len(assets):
                sort_order.append(node_index)
            else:
                left = int(linkage_matrix[node_index - len(assets), 0])
                right = int(linkage_matrix[node_index - len(assets), 1])
                recurse(left)
                recurse(right)

        recurse(len(linkage_matrix) + len(assets) - 2)
        sort_order = [i for i in sort_order if i < len(assets)]
        return [assets[i] for i in sort_order]

    @staticmethod
    def _get_recursive_bisection(cov: pd.DataFrame, sorted_assets: Iterable[str]) -> pd.Series:
        weights = pd.Series(1.0, index=sorted_assets)
        clusters = [list(sorted_assets)]
        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left_cluster = cluster[:split]
            right_cluster = cluster[split:]
            clusters.append(left_cluster)
            clusters.append(right_cluster)

            left_var = HRPAllocator._cluster_variance(cov, left_cluster)
            right_var = HRPAllocator._cluster_variance(cov, right_cluster)
            allocation = 1 - left_var / (left_var + right_var)
            weights[left_cluster] *= allocation
            weights[right_cluster] *= 1 - allocation
        return weights / weights.sum()

    @staticmethod
    def _cluster_variance(cov: pd.DataFrame, assets: Iterable[str]) -> float:
        cov_slice = cov.loc[assets, assets]
        inv_var = 1.0 / np.diag(cov_slice)
        weights = inv_var / inv_var.sum()
        variance = float(weights.T @ cov_slice @ weights)
        return variance


class BlackLittermanModel:
    """Black-Litterman blending of market equilibrium and investor views."""

    def __init__(self, tau: float = 0.05, risk_aversion: float = 2.5):
        self.tau = tau
        self.risk_aversion = risk_aversion

    def posterior(
        self,
        cov: pd.DataFrame,
        market_weights: pd.Series,
        market_views: Optional[pd.Series] = None,
        view_confidence: Optional[pd.Series] = None,
    ) -> pd.Series:
        symbols = cov.columns
        cov_matrix = cov.loc[symbols, symbols].values
        w_mkt = market_weights.reindex(symbols).fillna(0).values
        pi = self.risk_aversion * cov_matrix.dot(w_mkt)

        if market_views is None or market_views.empty:
            return pd.Series(pi, index=symbols)

        views = market_views.reindex(symbols).fillna(0).values
        active = market_views.dropna()
        P = np.eye(len(symbols))[np.isin(symbols, active.index)]
        Q = active.values
        if view_confidence is None:
            omega = np.diag(np.diag(P @ (self.tau * cov_matrix) @ P.T))
        else:
            confidence = view_confidence.reindex(active.index).fillna(1.0).values
            omega = np.diag(1.0 / confidence)

        tau_cov_inv = np.linalg.inv(self.tau * cov_matrix)
        middle = np.linalg.inv(tau_cov_inv + P.T @ np.linalg.inv(omega) @ P)
        posterior_mean = middle @ (tau_cov_inv @ pi + P.T @ np.linalg.inv(omega) @ Q)
        return pd.Series(posterior_mean, index=symbols)


@dataclass
class QuantumAnnealingConfig:
    iterations: int = 500
    temperature: float = 1.0
    cooling_rate: float = 0.995
    risk_aversion: float = 3.0
    return_weight: float = 1.0
    transaction_cost_penalty: float = 1.0


class QuantumAnnealingOptimizer:
    """Simulated annealing heuristic inspired by quantum annealing."""

    def __init__(self, config: QuantumAnnealingConfig):
        self.config = config

    def optimise(
        self,
        initial_weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        transaction_costs: Optional[np.ndarray] = None,
        current_weights: Optional[np.ndarray] = None,
        sector_map: Optional[np.ndarray] = None,
        sector_limits: Optional[Dict[str, float]] = None,
        regime_state: Optional[int] = None,
    ) -> np.ndarray:
        weights = initial_weights.copy()
        best_weights = weights.copy()
        risk_aversion = self.config.risk_aversion * (1.2 if regime_state and regime_state > 0 else 1.0)
        temperature = self.config.temperature
        best_objective = self._objective(
            weights, expected_returns, cov_matrix, risk_aversion, transaction_costs, current_weights
        )

        for _ in range(self.config.iterations):
            candidate = self._propose(weights)
            candidate = self._project(candidate, sector_map, sector_limits)
            objective = self._objective(
                candidate, expected_returns, cov_matrix, risk_aversion, transaction_costs, current_weights
            )
            delta = objective - best_objective
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                weights = candidate
                if objective < best_objective:
                    best_objective = objective
                    best_weights = candidate.copy()
            temperature *= self.config.cooling_rate
        return best_weights

    def _objective(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float,
        transaction_costs: Optional[np.ndarray],
        current_weights: Optional[np.ndarray],
    ) -> float:
        variance = weights @ cov_matrix @ weights
        ret = weights @ expected_returns
        cost = 0.0
        if transaction_costs is not None and current_weights is not None:
            cost = np.sum(transaction_costs * np.abs(weights - current_weights))
        return risk_aversion * variance - self.config.return_weight * ret + self.config.transaction_cost_penalty * cost

    @staticmethod
    def _propose(weights: np.ndarray) -> np.ndarray:
        perturbation = np.random.normal(0, 0.05, size=weights.shape[0])
        candidate = weights + perturbation
        candidate = np.clip(candidate, 0, None)
        if candidate.sum() == 0:
            candidate = np.ones_like(candidate) / len(candidate)
        else:
            candidate /= candidate.sum()
        return candidate

    @staticmethod
    def _project(
        weights: np.ndarray,
        sector_map: Optional[np.ndarray],
        sector_limits: Optional[Dict[str, float]],
    ) -> np.ndarray:
        if sector_map is None or sector_limits is None:
            return weights
        weights = weights.copy()
        unique_sectors = np.unique(sector_map)
        for sector in unique_sectors:
            limit = sector_limits.get(sector)
            if limit is None:
                continue
            mask = sector_map == sector
            sector_weight = weights[mask].sum()
            if sector_weight > limit:
                weights[mask] *= limit / sector_weight
        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= weights.sum()
        return weights


class PortfolioOptimizer:
    """High-level portfolio optimiser integrating multiple risk/return models."""

    def __init__(
        self,
        ewma_window: int = 252,
        ewma_decay: float = 0.97,
        hrp_enabled: bool = True,
        bl_tau: float = 0.05,
        risk_aversion: float = 3.0,
        sector_limits: Optional[Dict[str, float]] = None,
        annealing_config: Optional[Dict[str, float]] = None,
    ):
        self.cov_estimator = CovarianceEstimator(window=ewma_window, decay=ewma_decay)
        self.factor_model = FactorModelPCA()
        self.hrp_enabled = hrp_enabled
        self.bl_model = BlackLittermanModel(tau=bl_tau, risk_aversion=risk_aversion)
        self.sector_limits = sector_limits or {}
        config = QuantumAnnealingConfig(**(annealing_config or {}))
        config.risk_aversion = risk_aversion
        self.annealer = QuantumAnnealingOptimizer(config)

    def optimise(
        self,
        price_history: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
        transaction_costs: Optional[pd.Series] = None,
        sector_map: Optional[pd.Series] = None,
        market_views: Optional[pd.Series] = None,
        view_confidence: Optional[pd.Series] = None,
        regime_result: Optional[RegimeDetectionResult] = None,
    ) -> pd.Series:
        price_history = price_history.dropna(axis=1, how="any")
        returns = price_history.pct_change().dropna()
        if returns.empty:
            raise ValueError("Price history insufficient for optimisation.")

        cov = self.cov_estimator.estimate(returns)
        self.factor_model.fit(returns)

        if self.hrp_enabled:
            initial_weights = HRPAllocator.allocate(cov)
        else:
            initial_weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)

        market_weights = initial_weights.copy()
        posterior_returns = self.bl_model.posterior(cov, market_weights, market_views, view_confidence)

        transaction_cost_array = None
        current_weight_array = None
        if transaction_costs is not None:
            transaction_cost_array = transaction_costs.reindex(cov.columns).fillna(transaction_costs.mean()).values
        if current_weights is not None:
            current_weight_array = current_weights.reindex(cov.columns).fillna(0).values

        sector_array = None
        if sector_map is not None:
            sector_array = sector_map.reindex(cov.columns).fillna("UNSPECIFIED").values

        regime_state = None
        if regime_result is not None:
            latest_state = regime_result.states.iloc[-1]
            regime_state = int(latest_state)

        optimised_weights = self.annealer.optimise(
            initial_weights=initial_weights.reindex(cov.columns).values,
            expected_returns=posterior_returns.reindex(cov.columns).values,
            cov_matrix=cov.values,
            transaction_costs=transaction_cost_array,
            current_weights=current_weight_array,
            sector_map=sector_array,
            sector_limits=self.sector_limits,
            regime_state=regime_state,
        )

        weight_series = pd.Series(optimised_weights, index=cov.columns)
        weight_series /= weight_series.sum()
        return weight_series
