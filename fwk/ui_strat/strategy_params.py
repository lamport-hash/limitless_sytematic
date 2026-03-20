"""
Strategy Parameter Models.

Pydantic models for validating strategy parameters in the UI.
"""

from typing import List, Tuple, Optional
from pydantic import BaseModel, Field


class DualMomentumParams(BaseModel):
    """Parameters for Dual Momentum strategy."""
    
    lookback: int = Field(default=3500, ge=0, description="ROC calculation window (bars)")
    default_asset: str = Field(..., description="Asset to hold when no momentum (safe haven)")
    top_n: int = Field(default=1, ge=1, le=10, description="Number of top momentum assets to allocate to")
    abs_momentum_threshold: float = Field(default=0.0, description="Minimum ROC required for allocation")
    min_holding_periods: int = Field(default=240, ge=0, description="Minimum bars to hold before switch (0=disabled)")
    switch_threshold_pct: float = Field(default=0.0, ge=0, description="New asset must be this % better to switch")
    rsi_period: int = Field(default=14, ge=1, le=100, description="RSI calculation period")
    use_rsi_entry_filter: bool = Field(default=False, description="Enable RSI entry filter")
    rsi_entry_max: float = Field(default=30.0, ge=0, le=100, description="Max RSI to allow entry")
    use_rsi_entry_queue: bool = Field(default=False, description="Queue blocked assets for RSI entry")
    use_rsi_diff_filter: bool = Field(default=False, description="Enable RSI difference filter")
    rsi_diff_threshold: float = Field(default=10.0, ge=0, le=100, description="Min RSI diff to switch")


class CtoLineParams(BaseModel):
    """Parameters for CTO Line strategy."""
    
    cto_v1: int = Field(default=15, ge=1, le=100, description="Fast SMMA period")
    cto_m1: int = Field(default=19, ge=1, le=100, description="Medium SMMA 1 period")
    cto_m2: int = Field(default=25, ge=1, le=100, description="Medium SMMA 2 period")
    cto_v2: int = Field(default=29, ge=1, le=100, description="Slow SMMA period")
    direction: str = Field(default="both", description="Trading direction: long, short, or both")
    min_holding_periods: int = Field(default=0, ge=0, description="Minimum bars to hold before switch (0=disabled)")
    switch_threshold_pct: float = Field(default=0.0, ge=0, description="Not used in basket mode")
    default_asset: str = Field(..., description="Asset to hold when no signals (REQUIRED)")
    cap_to_half_assets: bool = Field(default=True, description="Cap allocations to at most half the assets")
    rsi_period: int = Field(default=14, ge=1, le=100, description="RSI calculation period")
    use_rsi_entry_filter: bool = Field(default=False, description="Enable RSI entry filter")
    rsi_entry_max: float = Field(default=30.0, ge=0, le=100, description="Max RSI to allow entry")
    use_rsi_entry_queue: bool = Field(default=False, description="Queue blocked assets for RSI entry")
    use_rsi_diff_filter: bool = Field(default=False, description="Enable RSI difference filter")
    rsi_diff_threshold: float = Field(default=10.0, ge=0, le=100, description="Min RSI diff to switch")
    
    @property
    def cto_params(self) -> Tuple[int, int, int, int]:
        """Return CTO params as tuple."""
        return (self.cto_v1, self.cto_m1, self.cto_m2, self.cto_v2)
    
    def validate_direction(self, v: str) -> str:
        if v not in ("long", "short", "both"):
            raise ValueError("direction must be 'long', 'short', or 'both'")
        return v


class BacktestRequest(BaseModel):
    """Request model for running a backtest."""
    
    filename: str = Field(..., description="Parquet file name")
    selected_assets: List[str] = Field(..., min_length=1, description="Assets to include in backtest")
    strategy_type: str = Field(default="dual_momentum", description="Strategy type: dual_momentum or cto_line")
    transaction_cost_pct: float = Field(default=0.01, ge=0, le=10, description="Transaction cost percentage")
    
    dual_momentum_params: Optional[DualMomentumParams] = None
    cto_line_params: Optional[CtoLineParams] = None


STRATEGY_TYPES = {
    "dual_momentum": {
        "display_name": "Dual Momentum",
        "description": "Classic dual momentum: ROC ranking + absolute momentum filter",
        "params_model": DualMomentumParams,
    },
    "cto_line": {
        "display_name": "CTO Line",
        "description": "Colored Trend Oscillator: SMMA crossover signals for basket allocation",
        "params_model": CtoLineParams,
    },
}


def get_strategy_params_schema(strategy_type: str) -> dict:
    """Get the parameter schema for a strategy type."""
    if strategy_type not in STRATEGY_TYPES:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    model_class = STRATEGY_TYPES[strategy_type]["params_model"]
    schema = {}
    
    for field_name, field_info in model_class.model_fields.items():
        schema[field_name] = {
            "type": str(field_info.annotation).replace("Optional[", "").replace("]", "").strip("'"),
            "default": field_info.default,
            "description": field_info.description or "",
            "required": field_info.is_required(),
        }
    
    return schema
