from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Paths(BaseSettings):
    ROOT: Path = Path(__file__).parent.parent
    DATA_RAW: Path = ROOT / "data" / "raw"
    DATA_PROCESSED: Path = ROOT / "data" / "processed"
    OUTPUTS: Path = ROOT / "outputs"
    FIGURES: Path = OUTPUTS / "figures"

    def create_dirs(self) -> None:
        for path in [
            self.DATA_RAW,
            self.DATA_PROCESSED,
            self.OUTPUTS,
            self.FIGURES,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class RedundantFeatureRule(BaseModel):
    feature_a: str
    feature_b: str
    drop: str
    threshold: float | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_drop(self) -> "RedundantFeatureRule":
        if self.drop not in {self.feature_a, self.feature_b}:
            raise ValueError("Поле 'drop' должно совпадать с feature_a или feature_b.")
        return self


class TargetEncodingFeature(BaseModel):
    name: str
    features: list[str]
    smoothing: float | None = None
    min_samples: int | None = None
    folds: int | None = None

    @model_validator(mode="after")
    def validate_features(self) -> "TargetEncodingFeature":
        if not self.features:
            raise ValueError("Нужно указать хотя бы одну колонку для target encoding.")
        return self


class DataConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA_")

    train_file: str = "train.csv"
    test_file: str = "test.csv"

    numeric_features: list[str] = Field(
        default_factory=lambda: [
            "person_age",
            "person_income",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
        ]
    )

    categorical_features: list[str] = Field(
        default_factory=lambda: [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]
    )

    target: str = "loan_status"
    index_col: int = 0


class OutlierConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OUTLIER_")

    min_work_age: int = 14
    max_emp_length: int = 50
    max_age: int = 90
    person_age_threshold: int = 123
    emp_length_threshold: float = 123.0
    max_stat_outlier_ratio: float = 0.05


class VisualizationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIZ_")

    theme: Literal["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"] = (
        "plotly_white"
    )
    width: int = 1200
    height: int = 600
    save_format: Literal["png", "svg", "pdf", "html"] = "html"

    categorical_colors: list[str] = Field(
        default_factory=lambda: [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
    )


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIPELINE_")

    consensus_threshold: int = 4
    feature_importance_estimators: int = 200
    permutation_importance_repeats: int = 3
    feature_select_top_n: int = 10
    feature_select_min_importance: float = 0.0
    parallel_coordinates_sample_size: int = 1000
    low_variance_threshold: float = 1e-4
    dominant_value_ratio: float = 0.99
    impossible_features: list[str] = Field(default_factory=list)
    importance_cv_splits: int = 5
    drop_redundant_features: bool = True
    redundant_feature_threshold: float = 0.85
    redundant_feature_rules: list[RedundantFeatureRule] = Field(default_factory=list)
    redundant_feature_discovery: bool = True
    redundant_feature_auto_drop: bool = True
    redundant_feature_report_name: str = "redundant_pairs.csv"
    redundant_feature_keep_priority: list[str] = Field(
        default_factory=lambda: [
            "cb_person_cred_hist_length",
            "loan_int_rate",
            "loan_amnt",
            "person_income",
        ]
    )
    redundant_feature_cv_splits: int = 5
    protected_features: list[str] = Field(
        default_factory=lambda: [
            "loan_amnt",
            "person_income",
            "loan_int_rate",
            "person_age",
        ]
    )
    feature_engineering_enabled: list[str] = Field(
        default_factory=lambda: [
            "ratios",
            "target_encoding",
        ]
    )
    drop_features_after_engineering: list[str] = Field(
        default_factory=lambda: [
            "loan_intent",
            "person_home_ownership",
            "loan_amnt",
            "cb_person_cred_hist_length",
        ]
    )
    target_encoding_folds: int = 5
    target_encoding_min_samples: int = 10
    target_encoding_smoothing: float = 20.0
    target_encoding_features: list[TargetEncodingFeature] = Field(
        default_factory=lambda: [
            TargetEncodingFeature(name="loan_intent_te", features=["loan_intent"]),
            TargetEncodingFeature(
                name="person_home_ownership_te", features=["person_home_ownership"]
            ),
        ]
    )
    auto_feature_discovery_enabled: bool = True
    auto_feature_numeric_pairs: int = 20
    auto_feature_categorical_pairs: int = 20
    auto_feature_top_k: int = 10
    auto_feature_min_auc: float = 0.55
    auto_feature_report_name: str = "auto_feature_candidates.csv"
    auto_feature_cv_splits: int = 5


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    paths: Paths = Field(default_factory=Paths)
    data: DataConfig = Field(default_factory=DataConfig)
    outlier: OutlierConfig = Field(default_factory=OutlierConfig)
    viz: VisualizationConfig = Field(default_factory=VisualizationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    random_seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


config = Config()
config.paths.create_dirs()
