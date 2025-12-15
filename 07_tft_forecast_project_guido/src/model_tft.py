from darts.models import TFTModel
from src.config import ModelConfig
import torch
from darts.utils.likelihood_models import QuantileRegression





def create_tft_model(model_cfg: ModelConfig,
                     input_chunk_length=None,
                     output_chunk_length=None,
                     use_quantile: bool = True, **kwargs) -> TFTModel:
    icl = input_chunk_length or model_cfg.input_chunk_length
    ocl = output_chunk_length or model_cfg.output_chunk_length

    # Configure likelihood / loss
    likelihood = None
    loss_fn = None
    if use_quantile:
        likelihood = QuantileRegression(quantiles=list(model_cfg.quantiles))
        loss_fn = None  # handled by likelihood
    else:
        likelihood = None
        loss_fn = torch.nn.MSELoss()

    model = TFTModel(
        input_chunk_length=icl,
        output_chunk_length=ocl,
        hidden_size=model_cfg.hidden_size,
        lstm_layers=model_cfg.lstm_layers,
        dropout=model_cfg.dropout,
        batch_size=model_cfg.batch_size,
        n_epochs=model_cfg.n_epochs,
        random_state=model_cfg.random_state,
        add_relative_index=model_cfg.add_relative_index,
        add_encoders=model_cfg.add_encoders,
        likelihood=likelihood,
        loss_fn=loss_fn,
        optimizer_cls=model_cfg.optimizer_class,
        optimizer_kwargs=model_cfg.optimizer_kwargs,
        lr_scheduler_cls=model_cfg.lr_scheduler_class,
        lr_scheduler_kwargs=model_cfg.lr_scheduler_kwargs,
        pl_trainer_kwargs=model_cfg.pl_trainer_kwargs,
    )
    return model



