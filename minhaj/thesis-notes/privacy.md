# Privacy and Federation

Federated learning keeps raw JIRA issue data inside the project that owns it. Clients train locally and send selected model updates to the server.

## Shared parameters

The server aggregates trainable LoRA-B matrices and categorical embeddings. In shared-head mode, the prediction head is aggregated too.

## Personalized mode

With `--personalized-head`, each project keeps its own prediction head. This helps account for team-specific interpretations of story point values while allowing the shared representation to learn from multiple projects.

## Scope of the privacy claim

The system prevents raw issue data from being exchanged. It does not by itself prevent attacks against transmitted model updates. Secure aggregation and differential privacy are potential future extensions.
