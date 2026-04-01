import numpy as np

from fl.client import FederatedClient
from fl.server import FedProxServer


def main() -> None:
    seed = 42
    rng = np.random.default_rng(seed)

    total_clients = 5
    per_round = 3
    rounds = 4

    selected_each_round = []
    for round_idx in range(rounds):
        picked = FedProxServer.choose_client_indices(total_clients=total_clients, per_round=per_round, rng=rng)
        picked_list = [int(x) for x in picked.tolist()]
        selected_each_round.append(picked_list)
        print(f"Round {round_idx + 1}: selected_clients={picked_list}")

    unique_round_patterns = {tuple(item) for item in selected_each_round}
    print(f"Client selection differs across rounds: {len(unique_round_patterns) > 1}")

    epoch_rng = np.random.default_rng(seed)
    num_examples = 12
    local_epochs = 4
    epoch_indices = []
    for epoch_idx in range(local_epochs):
        sampled = FederatedClient.sample_epoch_indices(
            num_examples=num_examples,
            sample_ratio_per_epoch=0.5,
            sample_with_replacement=False,
            rng=epoch_rng,
        )
        sampled_list = [int(x) for x in sampled.tolist()]
        epoch_indices.append(sampled_list)
        print(f"Epoch {epoch_idx + 1}: sampled_count={len(sampled_list)} indices={sampled_list}")

    unique_epoch_patterns = {tuple(item) for item in epoch_indices}
    print(f"Epoch sampling differs across epochs: {len(unique_epoch_patterns) > 1}")


if __name__ == "__main__":
    main()
