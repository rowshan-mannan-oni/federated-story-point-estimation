# Dataset Notes

The project uses the TAWOS dataset, a collection of JIRA issues from open-source software projects.

## Data shape

- One CSV file represents one software project.
- Each project is a federated learning client.
- Important fields include title, description, story point, type, priority, and creation date.
- Missing descriptions are retained as title-only examples.

## Labels

The task uses five ordinal story point classes:

`{1, 2, 3, 5, 8}`

Rows with unsupported story point values are filtered during preparation and recorded in the preprocessing report.
