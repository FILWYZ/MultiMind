# MultiMind

MultiMind is a customer-service agent runtime for e-commerce support. It classifies user requests, routes them to specialist agents, retrieves relevant knowledge when needed, and coordinates multi-agent responses. The platform also provides session memory, tool controls, runtime traces, and quality evaluation hooks.

## Core technologies

- Python and FastAPI for the service API
- LLM, embeddings, and rule-based intent recognition
- Multi-agent orchestration with specialist routing and response composition
- Retrieval-augmented generation with query rewriting, parallel retrieval, deduplication, and reranking
- Redis and ChromaDB-backed conversation and knowledge services
- Vue for the operator interface
- Docker for runtime packaging
