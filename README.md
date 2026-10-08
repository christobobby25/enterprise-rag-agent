# Enterprise RAG Agent

An AWS-deployed Retrieval-Augmented Generation (RAG) application that answers questions using enterprise documents, combining semantic search, LLM-based agents, and workflow orchestration.

Built with Python, Amazon Bedrock, AWS Strands Agents, LangGraph, FAISS, Amazon S3, and Amazon Bedrock AgentCore.

## Overview

Enterprise RAG Agent enables users to ask natural-language questions about PDF documents and receive answers grounded in retrieved document content.

The application uses LangGraph to route questions between general-purpose LLM responses and document-specific retrieval workflows. For document questions, a Strands agent uses a semantic search tool to retrieve relevant context before generating an answer with Amazon Nova Lite.

The application is deployed to Amazon Bedrock AgentCore Runtime, with AWS CDK used to define its deployment infrastructure.

## Architecture

```text
User Question
     |
     v
LangGraph Router
     |
     +------------------------+
     |                        |
     v                        v
General Question        Document Question
     |                        |
     v                        v
Amazon Bedrock          AWS Strands Agent
Nova Lite                     |
     |                        v
     |                  search_documents Tool
     |                        |
     |                        v
     |                  FAISS Vector Index
     |                        |
     |                        v
     |                  Retrieved PDF Chunks
     |                        |
     |                        v
     |                  Amazon Nova Lite
     |                        |
     +------------+-----------+
                  |
                  v
          Formatted Response
```

### Document ingestion pipeline

1. Load and extract text from PDF documents.
2. Split extracted content into smaller text chunks.
3. Generate vector embeddings using Amazon Titan Text Embeddings V2.
4. Store vectors in a FAISS index.
5. Persist the index in Amazon S3 for reuse.

### Question-answering workflow

1. Receive a natural-language question.
2. Route the request through LangGraph.
3. For document questions, invoke a Strands agent with a document-search tool.
4. Retrieve relevant chunks from FAISS.
5. Generate an answer using Amazon Nova Lite.
6. Format the response and include source references when available.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| LLMs | Amazon Nova Lite |
| Embeddings | Amazon Titan Text Embeddings V2 |
| Agent framework | AWS Strands Agents |
| Workflow orchestration | LangGraph |
| Vector search | FAISS |
| Storage | Amazon S3 |
| Cloud deployment | Amazon Bedrock AgentCore Runtime |
| Infrastructure as Code | AWS CDK, TypeScript |
| AWS integration | Boto3 |
| Testing | pytest, Ruff |
| CI | GitHub Actions |

## Key Features

### Retrieval-Augmented Generation

Uses semantic retrieval to supply relevant PDF content to the language model, enabling document-grounded answers instead of relying exclusively on the model's pretrained knowledge.

### Agent-Based Tool Calling

Integrates AWS Strands Agents with a custom `search_documents` tool, allowing the model to retrieve document context during question answering.

### Intelligent Question Routing

Uses LangGraph to distinguish general questions from document-related questions and direct each request to the appropriate workflow.

### Source-Aware Responses

Preserves document source information and page references to make retrieved information easier to trace.

### AWS Cloud Deployment

Runs on Amazon Bedrock AgentCore Runtime, with AWS CDK managing deployment infrastructure and Amazon S3 storing the FAISS index.

### Automated Testing and CI

Uses pytest for application testing and Ruff for linting. GitHub Actions validates Python code and compiles the AWS CDK TypeScript project on pushes and pull requests to `main`.

## Example

**Question:**

What is cloud computing according to the AWS Overview Whitepaper?

**Response:**

Cloud computing is the on-demand delivery of IT resources over the internet with pay-as-you-go pricing.

Source: `aws-overview.pdf`, page 12.

This example is based on a successful deployed runtime invocation.

## Testing

Run the automated test suite:

```bash
python -m pytest -q
```

Run linting:

```bash
python -m ruff check .
```

The project currently includes 40 passing automated tests.

## AWS Deployment

The application is deployed using Amazon Bedrock AgentCore Runtime.

Deployment components include:

- AgentCore Runtime for executing the application.
- AWS CDK for infrastructure configuration.
- Amazon Bedrock for model inference and embeddings.
- Amazon S3 for vector-index persistence.
- Boto3 for AWS service integration.

The CDK configuration requires the `RAG_S3_BUCKET` and `RAG_INDEX_KEY` environment variables.

Deployment requires appropriately configured AWS credentials, permissions, and access to the required Amazon Bedrock models.

## Current Limitations

- The initial retrieval corpus consists of the AWS Overview Whitepaper.
- Amazon Nova Lite has occasionally returned an invalid tool-use sequence during agent execution, causing intermittent request failures.
- Additional retry handling, retrieval evaluation, and operational monitoring would improve reliability.
- The current application is a portfolio demonstration rather than a fully hardened production service.

## Future Improvements

- Add retry and fallback handling for intermittent model tool-calling failures.
- Support multiple document collections and document updates.
- Expand retrieval and groundedness evaluation.
- Improve observability and operational monitoring.
- Add automated deployment workflows with secure AWS authentication.

## Project Purpose

This project was developed to gain practical experience designing, testing, and deploying GenAI applications on AWS, with emphasis on retrieval architecture, agent orchestration, infrastructure as code, and cloud-based execution.
