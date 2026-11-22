"""Server-side orchestration for SRA-FL simulation (Modules 1–7)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import flwr
import numpy as np
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate

from sra_fl.attacks.lying_injector import LyingInjector
from sra_fl.audit.audit_engine import AuditEngine
from sra_fl.client.client_app import FlowerClient
from sra_fl.client.client_guardian import ClientGuardian
from sra_fl.client.local_trainer import evaluate_model
from sra_fl.client.negotiation_handler import NegotiationHandler
from sra_fl.communication.trigger_policy import TriggerPolicy
from sra_fl.communication.upload_manager import UploadManager
from sra_fl.datasets.loader import load_federated_dataset
from sra_fl.evaluation.metrics import MetricsTracker, weighted_average
from sra_fl.models.cnn import create_model
from sra_fl.models.model_utils import get_parameters, set_parameters, train_eval_device
from sra_fl.negotiation.client_agent import ClientNegotiationAgent
from sra_fl.negotiation.privacy_manager import PrivacyManager
from sra_fl.negotiation.quantization import compress_parameters
from sra_fl.negotiation.server_agent import ServerNegotiationAgent
from sra_fl.orchestration.global_orchestrator import GlobalOrchestrator
from sra_fl.server.client_selector import ClientSelector
from sra_fl.simulation.client_environment import ClientEnvironment
from sra_fl.utils.config_loader import PROJECT_ROOT, load_config
from sra_fl.utils.llm_client import LLMClient
from sra_fl.utils.logger import get_logger, setup_logging
from sra_fl.utils.seed import set_seed

logger = get_logger(__name__)


def _sample_clients(
    num_clients: int,
    fraction: float,
    min_clients: int,
    rng: random.Random,
) -> list[int]:
    sample_size = max(min_clients, int(np.ceil(fraction * num_clients)))
    sample_size = min(sample_size, num_clients)
    return rng.sample(range(num_clients), k=sample_size)


def _centralized_evaluate(
    *,
    dataset_name: str,
    parameters: list[np.ndarray],
    test_dataset,
    batch_size: int,
    device,
) -> tuple[float, float, int]:
    model = create_model(dataset_name)
    set_parameters(model, parameters)
    return evaluate_model(model, test_dataset, batch_size=batch_size, device=device)


def _module_tag(config: dict[str, Any]) -> str:
    if bool(config.get("enable_privacy", False)):
        return "module7"
    if bool(config.get("enable_communication", False)):
        return "module6"
    if bool(config.get("enable_rag", False)):
        return "module5"
    if bool(config.get("enable_negotiation", False)):
        return "module4"
    if bool(config.get("enable_audit", False)):
        return "module3"
    if bool(config.get("enable_heterogeneity", True)):
        return "module2"
    return "module1"


def run_simulation(config: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    """Run Flower-style FL with optional Modules 2–7 features."""
    if config is None:
        config = load_config(overrides=overrides or None)
    elif overrides:
        config = {**config, **overrides}

    results_dir = Path(config.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    setup_logging(results_dir / "logs")
    set_seed(int(config["seed"]))
    rng = random.Random(int(config["seed"]))

    strategy_name = str(config["strategy"]).lower()
    if strategy_name not in {"fedavg", "fedprox"}:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    enable_heterogeneity = bool(config.get("enable_heterogeneity", True))
    enable_audit = bool(config.get("enable_audit", False)) and enable_heterogeneity
    enable_negotiation = bool(config.get("enable_negotiation", False)) and enable_heterogeneity
    enable_rag = bool(config.get("enable_rag", False)) and enable_heterogeneity
    enable_communication = bool(config.get("enable_communication", False))
    enable_privacy = bool(config.get("enable_privacy", False)) and enable_negotiation
    module_tag = _module_tag(
        {
            **config,
            "enable_audit": enable_audit,
            "enable_negotiation": enable_negotiation,
            "enable_rag": enable_rag,
            "enable_communication": enable_communication,
            "enable_privacy": enable_privacy,
        }
    )

    logger.info(
        "Starting %s | dataset=%s strategy=%s clients=%s rounds=%s "
        "hetero=%s audit=%s negotiation=%s rag=%s comm=%s privacy=%s",
        module_tag,
        config["dataset"],
        strategy_name,
        config["num_clients"],
        config["num_rounds"],
        enable_heterogeneity,
        enable_audit,
        enable_negotiation,
        enable_rag,
        enable_communication,
        enable_privacy,
    )

    fed_data = load_federated_dataset(
        dataset_name=str(config["dataset"]),
        num_clients=int(config["num_clients"]),
        partition=str(config.get("partition", "non_iid")),
        dirichlet_alpha=float(config.get("dirichlet_alpha", 0.3)),
        seed=int(config["seed"]),
    )

    tracker = MetricsTracker(results_dir=results_dir)
    selector = ClientSelector()
    device = train_eval_device(str(config.get("device", "cpu")))

    partition_sizes = [int(item["num_samples"]) for item in fed_data["partition_summary"]]
    for item in fed_data["partition_summary"]:
        logger.info("Client %s | samples=%s", item["client_id"], item["num_samples"])

    client_env: ClientEnvironment | None = None
    lying_injector: LyingInjector | None = None
    audit_engine: AuditEngine | None = None
    negotiator: NegotiationHandler | None = None
    privacy_manager: PrivacyManager | None = None
    orchestrator: GlobalOrchestrator | None = None
    upload_manager: UploadManager | None = None
    negotiation_logs: list[dict[str, Any]] = []
    audit_logs: list[dict[str, Any]] = []
    rag_logs: list[dict[str, Any]] = []
    privacy_logs: list[dict[str, Any]] = []

    if enable_heterogeneity:
        client_env = ClientEnvironment.create(
            num_clients=int(config["num_clients"]),
            partition_sizes=partition_sizes,
            seed=int(config["seed"]),
            seconds_per_round=float(config.get("seconds_per_round", 3600.0)),
            dropout_probability=float(config.get("dropout_probability", 0.05)),
        )

        if enable_audit:
            forced_lie = config.get("forced_lie_type")
            lying_injector = LyingInjector(
                dishonest_fraction=float(config.get("dishonest_fraction", 0.3)),
                lie_compute_factor=float(config.get("lie_compute_factor", 3.0)),
                lie_bandwidth_factor=float(config.get("lie_bandwidth_factor", 4.0)),
                seed=int(config["seed"]),
                forced_lie_type=forced_lie,
            )
            lying_injector.assign(list(range(int(config["num_clients"]))))
            client_env.lying_injector = lying_injector
            # Apply initial reports
            for state in client_env.states.values():
                lying_injector.apply_to_state(state, actually_available=True)

            audit_mode = str(config.get("audit_mode", "historical")).lower()
            ema_alpha = 1.0 if audit_mode == "basic" else 0.4
            audit_engine = AuditEngine(
                list(range(int(config["num_clients"]))),
                suspicion_threshold=float(config.get("suspicion_threshold", 0.35)),
                reputation_penalty=float(config.get("reputation_penalty", 0.25)),
                reputation_reward=float(config.get("reputation_reward", 0.05)),
                initial_reputation=float(config.get("initial_reputation", 1.0)),
                min_reputation=float(config.get("min_reputation", 0.05)),
            )
            audit_engine.detector.ema.alpha = ema_alpha
            if audit_mode == "basic":
                # One-shot check: no reputation learning for selection.
                config["use_reputation_selection"] = False

        if enable_privacy:
            privacy_manager = PrivacyManager(
                list(range(int(config["num_clients"]))),
                default_epsilon=float(config.get("privacy_epsilon_default", 8.0)),
                base_cost=float(config.get("privacy_base_cost", 0.35)),
                seed=int(config["seed"]),
            )

        if enable_negotiation:
            negotiator = NegotiationHandler(
                client_agent=ClientNegotiationAgent(
                    ClientGuardian(
                        weak_device_epochs=int(config.get("weak_device_epochs", 1)),
                        strong_device_epochs=int(config.get("strong_device_epochs", 2)),
                        weak_device_compression=int(config.get("weak_device_compression", 2)),
                        medium_device_compression=int(
                            config.get("medium_device_compression", 1)
                        ),
                        strong_device_compression=int(
                            config.get("strong_device_compression", 0)
                        ),
                        default_batch_size=int(config["batch_size"]),
                    )
                ),
                server_agent=ServerNegotiationAgent(
                    server_preferred_epochs=int(config.get("server_preferred_epochs", 2)),
                    min_local_epochs=int(config.get("min_local_epochs", 1)),
                    max_local_epochs=int(config.get("max_local_epochs", 3)),
                    default_batch_size=int(config["batch_size"]),
                    default_learning_rate=float(config["learning_rate"]),
                ),
                max_rounds=int(config.get("max_negotiation_rounds", 3)),
                privacy_manager=privacy_manager,
            )

        if enable_rag:
            orchestrator = GlobalOrchestrator(
                llm=LLMClient(enabled=bool(config.get("enable_llm", True))),
                enable_llm=bool(config.get("enable_llm", True)),
                mode=str(config.get("rag_mode", "hierarchical")),
            )

        if enable_communication:
            upload_manager = UploadManager(
                policy=TriggerPolicy(
                    epsilon=float(config.get("comm_epsilon", 0.15)),
                    alpha=float(config.get("comm_alpha", 0.3)),
                    seed=int(config["seed"]),
                    mode=str(config.get("comm_policy_mode", "learning")),
                    bandwidth_threshold_kbps=float(
                        config.get("comm_bandwidth_threshold_kbps", 100.0)
                    ),
                )
            )

        states_path = results_dir / "metrics" / f"{module_tag}_client_states.json"
        with states_path.open("w", encoding="utf-8") as handle:
            json.dump(client_env.states_as_dicts(), handle, indent=2)
        logger.info("Saved client states to %s", states_path)

    global_model = create_model(fed_data["dataset_name"])
    global_parameters = get_parameters(global_model)

    loss0, acc0, n0 = _centralized_evaluate(
        dataset_name=fed_data["dataset_name"],
        parameters=global_parameters,
        test_dataset=fed_data["test_dataset"],
        batch_size=int(config["batch_size"]),
        device=device,
    )
    tracker.log_round(0, loss=float(loss0), accuracy=float(acc0), extra={"num_test_samples": n0})
    logger.info("Round 0 centralized eval | loss=%.4f accuracy=%.4f", loss0, acc0)

    history_losses: list[tuple[int, float]] = [(0, float(loss0))]
    history_accuracy: list[tuple[int, float]] = [(0, float(acc0))]

    num_clients = int(config["num_clients"])
    num_rounds = int(config["num_rounds"])
    proximal_mu = float(config.get("proximal_mu", 0.01)) if strategy_name == "fedprox" else 0.0
    use_reputation_selection = bool(config.get("use_reputation_selection", True))

    for server_round in range(1, num_rounds + 1):
        logger.info("[ROUND %s]", server_round)

        dropouts: list[int] = []
        available_ids: list[int] = list(range(num_clients))
        reputation_scores = (
            audit_engine.reputation.scores
            if audit_engine is not None and use_reputation_selection
            else None
        )

        if client_env is not None:
            fit_part = client_env.select_participants(
                server_round,
                fraction=float(config["fraction_fit"]),
                min_clients=int(config["min_fit_clients"]),
                reputation_scores=reputation_scores,
            )
            available_ids = fit_part.available_ids
            fit_ids = fit_part.selected_ids

            if (
                orchestrator is not None
                and bool(config.get("rag_use_for_selection", True))
                and available_ids
            ):
                reputations = (
                    audit_engine.reputation.scores if audit_engine is not None else {}
                )
                privacy_remaining = (
                    privacy_manager.remaining_map() if privacy_manager is not None else {}
                )
                orchestrator.sync_from_environment(
                    client_env.states,
                    reputations=reputations,
                    privacy_remaining=privacy_remaining,
                    available_ids=available_ids,
                )
                decision = orchestrator.decide(
                    available_ids=available_ids,
                    target_count=max(
                        int(config["min_fit_clients"]),
                        len(fit_ids),
                    ),
                )
                # Prefer orchestrator ranking while keeping availability constraints.
                recommended = [
                    cid for cid in decision.recommended_client_ids if cid in available_ids
                ]
                if len(recommended) >= int(config["min_fit_clients"]):
                    fit_ids = recommended
                rag_logs.append({"round": server_round, **decision.to_dict()})
                logger.info(
                    "RAG selection | recommended=%s explanation=%s",
                    fit_ids,
                    decision.explanations[0][:160] if decision.explanations else "",
                )

            logger.info(
                "availability | online=%s offline=%s selected=%s",
                len(fit_part.available_ids),
                len(fit_part.unavailable_ids),
                fit_ids,
            )
            if len(fit_ids) < int(config["min_fit_clients"]):
                logger.warning("Round %s: insufficient available clients", server_round)
                tracker.log_round(
                    server_round,
                    loss=history_losses[-1][1],
                    accuracy=history_accuracy[-1][1],
                    fit_participants=0,
                    evaluate_participants=0,
                    extra={"skipped": True, "reason": "insufficient_available_clients"},
                )
                continue
        else:
            fit_ids = _sample_clients(
                num_clients=num_clients,
                fraction=float(config["fraction_fit"]),
                min_clients=int(config["min_fit_clients"]),
                rng=rng,
            )

        selector.record(
            server_round,
            phase="fit",
            selected_ids=fit_ids,
            available_ids=available_ids,
        )
        tracker.log_participation(server_round, fit_ids, phase="fit")

        # Per-client negotiated configs
        client_fit_configs: dict[int, dict[str, Any]] = {}
        for client_id in fit_ids:
            base_cfg = {
                "server_round": server_round,
                "local_epochs": int(config["local_epochs"]),
                "batch_size": int(config["batch_size"]),
                "learning_rate": float(config["learning_rate"]),
                "momentum": float(config.get("momentum", 0.9)),
                "proximal_mu": proximal_mu,
                "compression_level": int(config.get("default_compression_level", 0)),
            }
            if negotiator is not None and client_env is not None:
                state = client_env.get_state(client_id)
                reputation = (
                    audit_engine.reputation.get(client_id) if audit_engine else 1.0
                )
                agreed, transcript = negotiator.negotiate(
                    state,
                    reputation=reputation,
                    round_id=server_round,
                )
                base_cfg["local_epochs"] = int(agreed.local_epochs)
                base_cfg["compression_level"] = int(agreed.compression_level)
                if agreed.batch_size is not None:
                    base_cfg["batch_size"] = int(agreed.batch_size)
                if agreed.learning_rate is not None:
                    base_cfg["learning_rate"] = float(agreed.learning_rate)
                privacy_remaining = None
                if privacy_manager is not None:
                    privacy_remaining = privacy_manager.accounts[client_id].remaining
                    privacy_logs.append(
                        {
                            "round": server_round,
                            "client_id": client_id,
                            "remaining": privacy_remaining,
                            "agreed_epochs": agreed.local_epochs,
                        }
                    )
                negotiation_logs.append(
                    {
                        "round": server_round,
                        "client_id": client_id,
                        "agreed": agreed.to_dict(),
                        "transcript": [m.to_dict() for m in transcript],
                        "reputation": reputation,
                        "privacy_remaining": privacy_remaining,
                    }
                )
                logger.info(
                    "Negotiation client %s | epochs=%s compression=%s",
                    client_id,
                    agreed.local_epochs,
                    agreed.compression_level,
                )
            client_fit_configs[client_id] = base_cfg

        weight_results: list[tuple[list[np.ndarray], int]] = []
        fit_metrics: list[tuple[int, dict[str, float]]] = []
        cost_metrics: list[dict[str, float]] = []
        observed_train_seconds: dict[int, float] = {}

        for client_id in fit_ids:
            if client_env is not None and client_env.should_dropout(client_id, server_round):
                dropouts.append(client_id)
                logger.info("Client %s dropped out during round %s", client_id, server_round)
                continue

            fit_cfg = client_fit_configs[client_id]
            client = FlowerClient(
                client_id=client_id,
                dataset=fed_data["client_datasets"][client_id],
                dataset_name=fed_data["dataset_name"],
                train_config=config,
            )
            updated_params, num_examples, metrics = client.fit(global_parameters, fit_cfg)

            # Apply negotiated compression before aggregation.
            compression_level = int(fit_cfg.get("compression_level", 0))
            if compression_level > 0:
                updated_params = compress_parameters(updated_params, compression_level)
                metrics["compression_level"] = float(compression_level)

            costs = {
                "train_seconds": 0.0,
                "upload_seconds": 0.0,
                "bandwidth_kbps": 1000.0,
            }
            if client_env is not None:
                costs = client_env.simulate_round_costs(
                    client_id,
                    local_epochs=int(fit_cfg["local_epochs"]),
                    parameters=updated_params,
                )
                metrics = {**metrics, **costs}
                observed_train_seconds[client_id] = float(costs["train_seconds"])
                state = client_env.get_state(client_id)
                reported = state.reported
                if reported is not None:
                    metrics["reported_computation"] = float(reported.computation)
                    metrics["reported_bandwidth_kbps"] = float(reported.bandwidth_kbps)
                cost_metrics.append(costs)

            # Module 6: learn whether to upload / compress / wait / skip.
            if upload_manager is not None:
                action = upload_manager.decide(
                    client_id,
                    train_loss=float(metrics.get("train_loss", 1.0)),
                    bandwidth_kbps=float(costs.get("bandwidth_kbps", 1000.0)),
                )
                prev_acc = history_accuracy[-1][1] if history_accuracy else 0.0
                maybe_params, comm_metrics = upload_manager.apply(
                    client_id,
                    action,
                    updated_params,
                    compression_level=compression_level,
                    upload_seconds=float(costs.get("upload_seconds", 0.0)),
                    accuracy_delta=0.0 if prev_acc is None else 0.01,
                )
                metrics.update(
                    {
                        "comm_action_upload": 1.0 if comm_metrics["uploaded"] else 0.0,
                        "comm_reward": float(comm_metrics["comm_reward"]),
                        "compression_level": float(
                            comm_metrics["compression_level_applied"]
                        ),
                    }
                )
                if maybe_params is None:
                    logger.info(
                        "Comm policy skipped upload for client %s action=%s",
                        client_id,
                        action,
                    )
                    fit_metrics.append((int(num_examples), metrics))
                    continue
                updated_params = maybe_params

            weight_results.append((updated_params, int(num_examples)))
            fit_metrics.append((int(num_examples), metrics))

        if not weight_results:
            logger.warning("Round %s: all selected clients dropped out", server_round)
            tracker.log_round(
                server_round,
                loss=history_losses[-1][1],
                accuracy=history_accuracy[-1][1],
                fit_participants=0,
                evaluate_participants=0,
                extra={"skipped": True, "reason": "all_dropouts", "dropouts": dropouts},
            )
            continue

        aggregated = aggregate(weight_results)
        global_parameters = [np.asarray(layer) for layer in aggregated]
        agg_fit_metrics = weighted_average(fit_metrics)
        logger.info("aggregate_fit | metrics=%s | dropouts=%s", agg_fit_metrics, dropouts)

        # AAVL audit after observing behavior this round
        audit_summary = None
        if audit_engine is not None and client_env is not None:
            audited_ids = [int(m["client_id"]) for _, m in fit_metrics if "client_id" in m]
            if not audited_ids:
                audited_ids = list(observed_train_seconds.keys()) or list(fit_ids)
            audit_summary = audit_engine.audit_clients(
                server_round,
                client_env.states,
                audited_ids,
                observed_train_seconds=observed_train_seconds,
                local_epochs=int(config["local_epochs"]),
            )
            audit_logs.append(audit_summary.to_dict())

        if client_env is not None:
            eval_part = client_env.select_participants(
                server_round,
                fraction=float(config["fraction_evaluate"]),
                min_clients=int(config["min_evaluate_clients"]),
                reputation_scores=reputation_scores,
            )
            eval_ids = eval_part.selected_ids
            available_ids = eval_part.available_ids
        else:
            eval_ids = _sample_clients(
                num_clients=num_clients,
                fraction=float(config["fraction_evaluate"]),
                min_clients=int(config["min_evaluate_clients"]),
                rng=rng,
            )

        selector.record(
            server_round,
            phase="evaluate",
            selected_ids=eval_ids,
            available_ids=available_ids,
        )
        tracker.log_participation(server_round, eval_ids, phase="evaluate")

        eval_config = {
            "server_round": server_round,
            "batch_size": int(config["batch_size"]),
        }
        eval_metrics: list[tuple[int, dict[str, float]]] = []
        eval_losses: list[tuple[int, float]] = []
        for client_id in eval_ids:
            client = FlowerClient(
                client_id=client_id,
                dataset=fed_data["client_datasets"][client_id],
                dataset_name=fed_data["dataset_name"],
                train_config=config,
            )
            loss, num_examples, metrics = client.evaluate(global_parameters, eval_config)
            eval_metrics.append((int(num_examples), metrics))
            eval_losses.append((int(num_examples), float(loss)))

        if eval_losses:
            total = sum(n for n, _ in eval_losses)
            distributed_loss = sum(n * value for n, value in eval_losses) / max(total, 1)
            logger.info(
                "aggregate_evaluate | loss=%.4f metrics=%s",
                distributed_loss,
                weighted_average(eval_metrics),
            )

        loss, accuracy, num_test = _centralized_evaluate(
            dataset_name=fed_data["dataset_name"],
            parameters=global_parameters,
            test_dataset=fed_data["test_dataset"],
            batch_size=int(config["batch_size"]),
            device=device,
        )

        extra: dict[str, Any] = {
            "num_test_samples": num_test,
            "fit_metrics": agg_fit_metrics,
            "strategy": strategy_name,
            "dropouts": dropouts,
            "num_available": len(available_ids),
        }
        if cost_metrics:
            extra["mean_train_seconds"] = float(
                np.mean([c["train_seconds"] for c in cost_metrics])
            )
            extra["mean_upload_seconds"] = float(
                np.mean([c["upload_seconds"] for c in cost_metrics])
            )
            extra["mean_energy_joules"] = float(
                np.mean([c["energy_joules"] for c in cost_metrics])
            )
        if audit_summary is not None:
            extra["audit"] = {
                "precision": audit_summary.precision,
                "recall": audit_summary.recall,
                "f1": audit_summary.f1,
                "flagged": sum(1 for r in audit_summary.records if r.flagged),
            }
            extra["reputation"] = audit_engine.reputation.summary() if audit_engine else {}
        if rag_logs and rag_logs[-1].get("round") == server_round:
            extra["rag_recommended"] = rag_logs[-1].get("recommended_client_ids")
            extra["rag_explanation"] = (rag_logs[-1].get("explanations") or [""])[0][:240]
        if upload_manager is not None:
            extra["comm_uploads"] = sum(
                1 for h in upload_manager.history if h.get("uploaded")
            )
            extra["comm_decisions"] = len(upload_manager.history)
        if privacy_manager is not None:
            extra["privacy_mean_remaining"] = privacy_manager.summary()["mean_remaining"]
        if orchestrator is not None:
            extra["llm_usage"] = orchestrator.llm.usage_summary()

        tracker.log_round(
            server_round,
            loss=float(loss),
            accuracy=float(accuracy),
            fit_participants=len(weight_results),
            evaluate_participants=len(eval_ids),
            extra=extra,
        )
        history_losses.append((server_round, float(loss)))
        history_accuracy.append((server_round, float(accuracy)))
        logger.info(
            "Round %s centralized eval | loss=%.4f accuracy=%.4f",
            server_round,
            loss,
            accuracy,
        )

    _ = ndarrays_to_parameters(global_parameters)

    run_name = str(
        config.get("experiment_name")
        or config.get("run_name")
        or (
            f"{module_tag}_{config['dataset']}_{strategy_name}_"
            f"c{config['num_clients']}_r{config['num_rounds']}"
        )
    )
    metrics_path = tracker.save(run_name=run_name)

    if audit_logs:
        audit_path = results_dir / "metrics" / f"{run_name}_audit.json"
        with audit_path.open("w", encoding="utf-8") as handle:
            json.dump(audit_logs, handle, indent=2)
        logger.info("Saved audit logs to %s", audit_path)
    if negotiation_logs:
        nego_path = results_dir / "metrics" / f"{run_name}_negotiation.json"
        with nego_path.open("w", encoding="utf-8") as handle:
            json.dump(negotiation_logs, handle, indent=2)
        logger.info("Saved negotiation logs to %s", nego_path)
    if rag_logs:
        rag_path = results_dir / "metrics" / f"{run_name}_rag.json"
        with rag_path.open("w", encoding="utf-8") as handle:
            json.dump(rag_logs, handle, indent=2)
        logger.info("Saved RAG logs to %s", rag_path)
    if upload_manager is not None and upload_manager.history:
        comm_path = results_dir / "metrics" / f"{run_name}_communication.json"
        with comm_path.open("w", encoding="utf-8") as handle:
            json.dump(upload_manager.history, handle, indent=2)
        logger.info("Saved communication logs to %s", comm_path)
    if privacy_manager is not None:
        priv_path = results_dir / "metrics" / f"{run_name}_privacy.json"
        with priv_path.open("w", encoding="utf-8") as handle:
            json.dump(privacy_manager.summary(), handle, indent=2)
        logger.info("Saved privacy logs to %s", priv_path)
    if orchestrator is not None:
        llm_path = results_dir / "metrics" / f"{run_name}_llm_usage.json"
        with llm_path.open("w", encoding="utf-8") as handle:
            json.dump(orchestrator.llm.usage_summary(), handle, indent=2)
        logger.info("Saved LLM usage to %s", llm_path)

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Flower version: %s", flwr.__version__)

    final_accuracy = history_accuracy[-1][1] if history_accuracy else None

    return {
        "history": {
            "losses_centralized": history_losses,
            "metrics_centralized": {"accuracy": history_accuracy},
        },
        "metrics_path": str(metrics_path),
        "tracker": tracker,
        "selector": selector,
        "partition_summary": fed_data["partition_summary"],
        "final_accuracy": final_accuracy,
        "config": config,
        "global_parameters": global_parameters,
        "client_environment": client_env,
        "audit_engine": audit_engine,
        "negotiation_logs": negotiation_logs,
        "audit_logs": audit_logs,
        "rag_logs": rag_logs,
        "privacy_manager": privacy_manager,
        "upload_manager": upload_manager,
        "orchestrator": orchestrator,
    }


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run SRA-FL Flower simulation")
    parser.add_argument("--dataset", default=None, help="fashion_mnist | cifar10")
    parser.add_argument("--strategy", default=None, help="fedavg | fedprox")
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--partition", default=None, help="iid | non_iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--no-heterogeneity", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--no-negotiation", action="store_true")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--no-communication", action="store_true")
    parser.add_argument("--no-privacy", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--dishonest-fraction", type=float, default=None)
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.strategy:
        overrides["strategy"] = args.strategy
    if args.num_clients is not None:
        overrides["num_clients"] = args.num_clients
    if args.num_rounds is not None:
        overrides["num_rounds"] = args.num_rounds
    if args.partition:
        overrides["partition"] = args.partition
    if args.dirichlet_alpha is not None:
        overrides["dirichlet_alpha"] = args.dirichlet_alpha
    if args.local_epochs is not None:
        overrides["local_epochs"] = args.local_epochs
    if args.no_heterogeneity:
        overrides["enable_heterogeneity"] = False
    if args.no_audit:
        overrides["enable_audit"] = False
    if args.no_negotiation:
        overrides["enable_negotiation"] = False
    if args.no_rag:
        overrides["enable_rag"] = False
    if args.no_communication:
        overrides["enable_communication"] = False
    if args.no_privacy:
        overrides["enable_privacy"] = False
    if args.no_llm:
        overrides["enable_llm"] = False
    if args.dishonest_fraction is not None:
        overrides["dishonest_fraction"] = args.dishonest_fraction

    dataset = overrides.get("dataset")
    config = load_config(dataset=dataset, overrides=overrides or None)
    run_simulation(config)


if __name__ == "__main__":
    main()
