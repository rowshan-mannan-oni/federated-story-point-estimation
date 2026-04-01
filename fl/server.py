from typing import Dict, List, Tuple
import time

import numpy as np
import torch

from fl.client import FederatedClient


class FedProxServer:
    """FedProx coordinator for deep learning models."""

    def __init__(self, model_factory, clients: List[FederatedClient], random_state: int) -> None:
        self.model_factory = model_factory
        self.clients = clients
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def choose_client_indices(total_clients: int, per_round: int, rng: np.random.Generator) -> np.ndarray:
        all_indices = np.arange(total_clients)
        return rng.choice(all_indices, size=per_round, replace=False)

    @staticmethod
    def _weighted_average_states(
        states: List[Dict[str, torch.Tensor]],
        weights: List[int],
    ) -> Dict[str, torch.Tensor]:
        total = float(sum(weights))
        avg_state: Dict[str, torch.Tensor] = {}

        for key in states[0].keys():
            first_tensor = states[0][key]

            # Float tensors are averaged. Non-float tensors are copied from the first client.
            if first_tensor.is_floating_point():
                combined = torch.zeros_like(first_tensor)
                for state, weight in zip(states, weights):
                    combined += state[key] * (weight / total)
                avg_state[key] = combined
            else:
                avg_state[key] = first_tensor.clone()

        return avg_state

    def train(
        self,
        rounds: int,
        clients_per_round_fraction: float,
        local_epochs: int,
        local_sample_ratio_per_epoch: float,
        sample_with_replacement: bool,
        learning_rate: float,
        weight_decay: float,
        prox_mu: float,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], List[float]]:
        global_model = self.model_factory().to(device)
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        history: List[float] = []

        per_round = max(1, int(round(len(self.clients) * clients_per_round_fraction)))

        print(
            f"[FedProx] Starting training: rounds={rounds}, total_clients={len(self.clients)}, "
            f"clients_per_round={per_round}, prox_mu={prox_mu}, "
            f"local_sample_ratio_per_epoch={local_sample_ratio_per_epoch}, "
            f"sample_with_replacement={sample_with_replacement}",
            flush=True,
        )

        train_start = time.perf_counter()

        for round_idx in range(rounds):
            round_start = time.perf_counter()
            picked = self.choose_client_indices(total_clients=len(self.clients), per_round=per_round, rng=self.rng)
            selected_client_ids = [self.clients[int(idx)].client_id for idx in picked]

            print(
                f"[FedProx][Round {round_idx + 1}/{rounds}] selected_clients={selected_client_ids}",
                flush=True,
            )

            client_states: List[Dict[str, torch.Tensor]] = []
            client_weights: List[int] = []
            client_losses: List[float] = []

            for idx in picked:
                client_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
                result = self.clients[int(idx)].train_local(
                    global_state=global_state,
                    model_factory=self.model_factory,
                    device=device,
                    epochs=local_epochs,
                    sample_ratio_per_epoch=local_sample_ratio_per_epoch,
                    sample_with_replacement=sample_with_replacement,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    prox_mu=prox_mu,
                    seed=client_seed,
                )
                client_states.append(result.state_dict)
                client_weights.append(result.num_examples)
                client_losses.append(result.loss)

                print(
                    f"  - client={self.clients[int(idx)].client_id} "
                    f"examples={result.num_examples} local_loss={result.loss:.6f}",
                    flush=True,
                )

            global_state = self._weighted_average_states(client_states, client_weights)
            round_mean_loss = float(np.mean(client_losses))
            weighted_round_loss = float(np.average(client_losses, weights=client_weights))
            history.append(round_mean_loss)

            completed_rounds = round_idx + 1
            round_elapsed = time.perf_counter() - round_start
            total_elapsed = time.perf_counter() - train_start
            avg_round_time = total_elapsed / completed_rounds
            remaining_rounds = max(rounds - completed_rounds, 0)
            eta_seconds = avg_round_time * remaining_rounds

            bar_width = 20
            filled = int((completed_rounds / max(rounds, 1)) * bar_width)
            bar = "#" * filled + "." * (bar_width - filled)

            print(
                f"[FedProx][Round {round_idx + 1}/{rounds}] "
                f"mean_local_loss={round_mean_loss:.6f} weighted_local_loss={weighted_round_loss:.6f}",
                flush=True,
            )
            print(
                f"[FedProx][Progress] [{bar}] {completed_rounds}/{rounds} "
                f"round_time={round_elapsed:.1f}s elapsed={total_elapsed:.1f}s eta={eta_seconds:.1f}s",
                flush=True,
            )

        return global_state, history
