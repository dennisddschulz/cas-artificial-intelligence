import optuna
import torch
from darts.utils.likelihood_models import QuantileRegression, GaussianLikelihood
from src.config import ExperimentConfig
from src.train_tft import run_experiment

def objective(trial: optuna.Trial):
    # Basis-Konfiguration laden
    cfg = ExperimentConfig()

    # --- Modellparameter
    cfg.model.hidden_size = trial.suggest_int("hidden_size", 16, 64)
    cfg.model.lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    cfg.model.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    cfg.model.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    cfg.model.n_epochs = 10  # kurz halten für Tuning

    # --- Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    cfg.model.optimizer_class = getattr(torch.optim, optimizer_name)
    cfg.model.optimizer_kwargs = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

    # --- Scheduler
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    if use_scheduler:
        from torch.optim.lr_scheduler import StepLR
        cfg.model.lr_scheduler_class = StepLR
        cfg.model.lr_scheduler_kwargs = {
            "step_size": trial.suggest_int("step_size", 5, 20),
            "gamma": trial.suggest_float("gamma", 0.1, 0.9),
        }
    else:
        cfg.model.lr_scheduler_class = None
        cfg.model.lr_scheduler_kwargs = {}

    # --- Likelihood
    likelihood_choice = trial.suggest_categorical("likelihood", ["none", "quantile", "gaussian"])
    if likelihood_choice == "quantile":
        cfg.model.likelihood = QuantileRegression(quantiles=[0.1, 0.5, 0.9])
    elif likelihood_choice == "gaussian":
        cfg.model.likelihood = GaussianLikelihood()
    else:
        cfg.model.likelihood = None

    # --- Weitere Encoder-Features
    cfg.model.add_relative_index = trial.suggest_categorical("add_relative_index", [True, False])
    encoder_type = trial.suggest_categorical("encoder_type", ["none", "dayofweek", "month"])
    cfg.model.add_encoders = None if encoder_type == "none" else {
        "datetime_attribute": {"past": [encoder_type]}
    }

    try:
        results = run_experiment(cfg)
        score = results["smape"]  # Minimieren

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    except Exception as e:
        print(f"⚠️ Trial abgebrochen: {e}")
        raise optuna.TrialPruned()


def run_optuna_search(n_trials: int = 30):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("✅ Beste Parameter:", study.best_params)
    print("📉 Bester SMAPE:", study.best_value)
    return study


if __name__ == "__main__":
    run_optuna_search(n_trials=20)
