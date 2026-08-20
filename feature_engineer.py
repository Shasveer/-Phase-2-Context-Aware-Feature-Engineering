"""Context-aware feature engineering for the cleaned customer dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create business-intelligent features with ethical safeguards."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_metadata: dict[str, str] = {}
        self._engineered_features: list[str] = []
        self._removed_features: list[str] = []
        self._max_vif = 1.0
        self._preserved_attrs = dict(getattr(df, "attrs", {}))
        self.train_stats: pd.DataFrame | None = None

    @staticmethod
    def _column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
        return next((name for name in names if name in df.columns), None)

    def _add_feature(self, name: str, values: Any, rationale: str) -> None:
        self.df[name] = values
        self._engineered_features.append(name)
        self.feature_metadata[name] = rationale

    def create_financial_strain_ratio(self) -> "FeatureEngineer":
        """Calculate debt-to-income safely, including customers with no income."""
        debt_column = self._column(self.df, ("debt", "monthly_debt", "total_debt"))
        income_column = self._column(
            self.df, ("income", "monthly_income", "annual_income")
        )
        if debt_column is None or income_column is None:
            self.feature_metadata["financial_strain_ratio"] = (
                "Not created: debt and income columns are required."
            )
            return self

        debt = pd.to_numeric(self.df[debt_column], errors="coerce").fillna(0)
        income = pd.to_numeric(self.df[income_column], errors="coerce").fillna(0)
        ratio = np.divide(
            debt.to_numpy(dtype=float),
            income.to_numpy(dtype=float),
            out=np.zeros(len(self.df), dtype=float),
            where=income.to_numpy(dtype=float) > 0,
        )
        self._add_feature(
            "financial_strain_ratio",
            ratio,
            "Identifies customers at risk of default due to overcommitment; zero income yields a neutral zero rather than infinity.",
        )
        return self

    def encode_load_shedding_impact(self) -> "FeatureEngineer":
        """Encode load-shedding hours on a weekly cycle."""
        column = self._column(self.df, ("load_shedding_hours", "load_shedding"))
        if column is None:
            self.feature_metadata["load_shedding_sin"] = (
                "Not created: load_shedding_hours is required."
            )
            return self

        hours = pd.to_numeric(self.df[column], errors="coerce").fillna(0)
        phase = 2 * np.pi * hours.to_numpy(dtype=float) / 168.0
        rationale = (
            "Preserves circular weekly timing so Friday 22h remains close to Saturday 02h, "
            "while representing infrastructure-related transaction risk without imposing a linear effect."
        )
        self._add_feature("load_shedding_sin", np.sin(phase), rationale)
        self._add_feature("load_shedding_cos", np.cos(phase), rationale)
        return self

    def regional_benchmarks(self, train_stats: Any = None) -> "FeatureEngineer":
        """Add regional income benchmarks from training data only."""
        region_column = self._column(self.df, ("region", "province", "location"))
        value_column = self._column(
            self.df, ("income", "monthly_income", "annual_income", "transaction_amount")
        )
        if region_column is None or value_column is None:
            self.feature_metadata["regional_income_benchmark"] = (
                "Not created: a region and numeric income column are required."
            )
            return self

        if train_stats is None:
            source = self.df[[region_column, value_column]].copy()
            source[value_column] = pd.to_numeric(source[value_column], errors="coerce")
            stats = source.groupby(region_column, dropna=False)[value_column].agg(
                regional_income_benchmark="mean", regional_sample_count="count"
            ).reset_index()
        elif isinstance(train_stats, Mapping):
            rows = []
            for region, values in train_stats.items():
                if isinstance(values, Mapping):
                    benchmark = next(
                        (values[key] for key in (
                            "regional_income_benchmark", "mean_income", "income_mean", "mean"
                        ) if key in values), None
                    )
                    sample_count = next(
                        (values[key] for key in (
                            "regional_sample_count", "sample_count", "count", "n"
                        ) if key in values), None
                    )
                else:
                    benchmark = values
                    sample_count = None
                row = {region_column: region, "regional_income_benchmark": benchmark}
                if sample_count is not None:
                    row["regional_sample_count"] = sample_count
                rows.append(row)
            stats = pd.DataFrame(rows)
        else:
            stats = pd.DataFrame(train_stats).copy()

        benchmark_column = "regional_income_benchmark"
        count_column = "regional_sample_count"
        required = {region_column, benchmark_column}
        if not required.issubset(stats.columns):
            raise ValueError(
                "train_stats must contain the region and regional_income_benchmark columns"
            )
        self.train_stats = stats.copy()
        merge_columns = [region_column, benchmark_column]
        if count_column in stats.columns:
            merge_columns.append(count_column)
        self.df = self.df.merge(stats[merge_columns], on=region_column, how="left", sort=False)
        self._engineered_features.append(benchmark_column)
        self.feature_metadata[benchmark_column] = (
            "Training-only regional income mean, used to contextualise financial behaviour without leaking validation or test observations."
        )
        if count_column in stats.columns:
            self._engineered_features.append(count_column)
            self.feature_metadata[count_column] = (
                "Training-only regional sample count; flags thin regional evidence and prevents overconfident comparisons."
            )
            small_regions = stats.loc[stats[count_column] < 50, region_column].tolist()
            if small_regions:
                self.feature_metadata["regional_benchmark_warning"] = (
                    f"Regions with fewer than 50 training samples: {small_regions}"
                )
        return self

    def encode_support_tickets(self) -> "FeatureEngineer":
        """Represent both whether support was needed and its positive volume."""
        column = "support_tickets" if "support_tickets" in self.df.columns else None
        if column is None:
            return self
        tickets = pd.to_numeric(self.df[column], errors="coerce").fillna(0).clip(lower=0)
        self._add_feature(
            "support_tickets_zero_flag",
            (tickets > 0).astype(int),
            "Separates the business meaning of no support need from the intensity of support demand.",
        )
        self._add_feature(
            "support_tickets_log1p",
            np.log1p(tickets),
            "Log1p compresses the zero-inflated support volume while retaining differences among active cases.",
        )
        return self

    def validate_multicollinearity(self, threshold: float = 0.85) -> "FeatureEngineer":
        """Drop later engineered features that duplicate earlier numeric features."""
        protected = {
            "financial_strain_ratio",
            "load_shedding_sin",
            "load_shedding_cos",
            "regional_income_benchmark",
            "support_tickets_zero_flag",
            "support_tickets_log1p",
        }
        candidates = [
            name for name in self._engineered_features if name in self.df and pd.api.types.is_numeric_dtype(self.df[name])
        ]
        retained: list[str] = []
        for name in candidates:
            if name not in protected and any(
                abs(self.df[[name, other]].corr().iloc[0, 1]) > threshold
                for other in retained
            ):
                self.df.drop(columns=name, inplace=True)
                self._removed_features.append(name)
                self.feature_metadata[name] += " Removed because its correlation exceeded the multicollinearity threshold."
            else:
                retained.append(name)
        self._engineered_features = [name for name in self._engineered_features if name not in self._removed_features]
        while len(retained) > 1:
            self._max_vif = self._calculate_max_vif(retained)
            if np.isfinite(self._max_vif) and self._max_vif <= 5.0:
                break
            vif_values = self._vif_values(retained)
            removable = [
                (index, name)
                for index, name in enumerate(retained)
                if name not in protected
            ]
            if not removable:
                break
            remove_index = max(removable, key=lambda item: vif_values[item[0]])[0]
            remove_name = retained[remove_index]
            self.df.drop(columns=remove_name, inplace=True)
            self._removed_features.append(remove_name)
            self.feature_metadata[remove_name] += " Removed because its VIF exceeded 5.0."
            retained.remove(remove_name)
            self._engineered_features.remove(remove_name)
        self._max_vif = self._calculate_max_vif(retained)
        return self

    def _vif_values(self, columns: list[str]) -> np.ndarray:
        values = self.df[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        correlation = values.corr().to_numpy()
        try:
            return np.diag(np.linalg.pinv(correlation))
        except (ValueError, np.linalg.LinAlgError):
            return np.full(len(columns), np.inf)

    def _calculate_max_vif(self, columns: list[str]) -> float:
        if len(columns) < 2:
            return 1.0
        if len(self.df) <= len(columns) + 1:
            return 1.0
        return float(np.nanmax(self._vif_values(columns)))

    def run_full_engineering(self) -> pd.DataFrame:
        """Execute all feature engineering steps and restore source metadata."""
        try:
            self.create_financial_strain_ratio()
            self.encode_load_shedding_impact()
            self.regional_benchmarks()
            self.encode_support_tickets()
            self.validate_multicollinearity()
            self.df.attrs.update(self._preserved_attrs)
            self.df.attrs["feature_metadata"] = dict(self.feature_metadata)
            return self.df
        except Exception as error:
            raise ValueError(f"Feature engineering failed: {error}") from error

    def __str__(self) -> str:
        return f"Created {len(self._engineered_features)} new features | Max VIF: {self._max_vif:.1f}"


def main() -> None:
    """Engineer the repository's processed customer file."""
    root = Path(__file__).resolve().parent
    input_path = root / "data" / "processed" / "cleaned_customers.csv"
    output_path = root / "data" / "processed" / "engineered_features.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Required Milestone 1 input not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineer = FeatureEngineer(pd.read_csv(input_path))
    engineered = engineer.run_full_engineering()
    engineered.to_csv(output_path, index=False)
    print(engineer)


if __name__ == "__main__":
    main()