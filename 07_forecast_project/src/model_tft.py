from darts.models import TFTModel
from src.config import ModelConfig
import torch


def create_tft_model(
    model_cfg: ModelConfig,
    input_chunk_length: int | None = None,
    output_chunk_length: int | None = None,
) -> TFTModel:
    """
    Creates a TFTModel with parameters from ModelConfig.
    """
    icl = input_chunk_length or model_cfg.input_chunk_length
    ocl = output_chunk_length or model_cfg.output_chunk_length
    print(model_cfg.n_epochs)
    model = TFTModel(
        input_chunk_length=icl,
        output_chunk_length=ocl,
        hidden_size=model_cfg.hidden_size,
        lstm_layers=model_cfg.lstm_layers,
        dropout=model_cfg.dropout,
        batch_size=model_cfg.batch_size,
        n_epochs=model_cfg.n_epochs,
        add_relative_index=True,
        add_encoders=None,  # students can experiment here later
        random_state=model_cfg.random_state,
        likelihood=None,    # students can play with QuantileRegression etc.
        loss_fn=torch.nn.MSELoss(),
        optimizer_kwargs={"lr": 1e-4},
        pl_trainer_kwargs={"gradient_clip_val": 0.1},
    )
    return model
