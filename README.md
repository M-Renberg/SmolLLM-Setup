# SmolLLM-Setup

A lightweight, production-ready asynchronous API framework for building custom LLM workflows and runnable sequences. Inspired by LangChain's Expression Language (LCEL), this project implements a clean `Runnable` pipeline interface and exposes it via a fast **FastAPI** web service.

## Features

- **FastAPI Endpoint**: Ready-to-use API routing for real-time ticket ingestion and processing.
- **Pydantic v2 Validation**: Full implementation of generic models (`Generic[I, O]`) ensuring strict input/output validation at every endpoint level.
- **Composable Pipelines**: Seamlessly chain analytical layers, data parsers, and custom routing logic together using the native OR (`|`) operator.
- **Lightweight Framework**: Minimal overhead infrastructure capable of routing and classifying raw support tickets into structured payloads.

---

## Installation & Running

This project is built and optimized using `uv` for ultra-fast dependency and environment management.

### 1. Install Dependencies
```bash
# Clone the repository
git clone [https://github.com/M-Renberg/SmolLLM-Setup.git](https://github.com/M-Renberg/SmolLLM-Setup.git)
`cd SmolLLM-Setup`

# Add required ecosystem packages
`uv add fastapi uvicorn transformers torch pydantic`

2. Start the FastAPI Server

Run the application using Uvicorn:
Bash

`uv run uvicorn main:app --reload`

The server will start spinning at http://127.0.0.1:8000. You can explore the interactive API docs at http://127.0.0.1:8000/docs.
API Reference
Process Ticket

    URL: /llm

    Method: POST

    Headers: Content-Type: application/json

Request Body Example
JSON

{
  "id": 1337,
  "message": "The payment portal is broken! Urgent! Fix ASAP"
}

Response Example (200 OK)
JSON

{
  "status": "routed",
  "assigned_to": "engineering_team",
  "ticket_details": {
    "customer_id": 1337,
    "sentiment": "negative",
    "urgency": "high",
    "summary": "The payment portal is broken! Urgent! Fix..."
    }
}

Pipeline Architecture

When a request hits the /llm endpoint, it runs through a sequential pipeline orchestrated behind the scenes:
Plaintext

[HTTP POST /llm] 
       │
       ▼
 ┌───────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
 │ LLMRequest│ ───> │TicketInput   │ ───> │Sentiment     │ ───> │TicketParser  │
 │ (FastAPI) │      │ (DTO Model)  │      │ Analyser     │      │ (Dict->Model)│
 └───────────┘      └──────────────┘      └──────────────┘      └──────────────┘
                                                                       │
                                                                       ▼
                                                             ┌──────────────────┐
                                                             │ route_ticket     │
                                                             │ (RunnableLambda) │
                                                             └──────────────────┘
                                                                       │
                                                                       ▼
                                                             [JSON Routed Response]

    SentimentAnalyser: Examines the incoming message to compute sentiment tags, urgency levels, and slices a short summary.

    TicketParser: Converts the raw dictionary into a strictly typed ProcessedTicket Pydantic structure.

    route_ticket: Evaluates urgency levels and assigns the final internal department destination.

Project Structure
Plaintext

├── main.py              # FastAPI application server & routing
└── llm/
    └── llm.py           # Core LCEL framework, schemas, and ticket_pipeline

License

This project is open-source and available under the MIT License.