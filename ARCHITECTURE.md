# Architecture Overview

## Layers

- `routers` -> HTTP layer with FastAPI endpoints only
- `services` -> business logic and orchestration
- `ml` -> ML inference and training components
- `utils` -> pure helper functions without business logic

## Modules Purpose

- `auth` -> JWT authentication and security flows
- `tasks` -> task CRUD and task lifecycle management
- `nlp` -> embeddings, semantic search, and RAG flows
- `cv` -> image classification, detection, segmentation, and embeddings
- `recsys` -> recommendation pipelines
- `celery` -> background task execution
- `monitoring` -> metrics, observability, and health checks

## Layer Rules

- Routers must not contain business logic.
- Services must not depend on HTTP concepts.
- ML code must stay isolated from API routing concerns.
- Utils must not work with databases or external services.
