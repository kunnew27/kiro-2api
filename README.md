# Kiro-2API

OpenAI compatible API gateway for Kiro (AWS CodeWhisperer AI).

Based on the KiroGate project, restructured following FastAPI best practices.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI API
- **Streaming Support**: Real-time streaming responses
- **Tool Calling**: Full support for function calling/tools
- **Image Support**: Vision capabilities with base64 images
- **Multi-tenant Mode**: Support for multiple users with separate tokens
- **Auto Token Refresh**: Automatic credential management
- **Adaptive Timeouts**: Smart timeout handling for slow models (Opus)

## Project Structure

```
kiro-2api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── core/
│   │   ├── config.py           # Settings and configuration
│   │   └── exceptions.py       # Exception handlers
│   ├── models/
│   │   └── schemas.py          # Pydantic models (OpenAI format)
│   ├── routes/
│   │   ├── chat.py             # /v1/chat/completions
│   │   ├── models.py           # /v1/models
│   │   └── health.py           # Health check endpoints
│   ├── middleware/
│   │   └── tracking.py         # Request tracking middleware
│   ├── libs/
│   │   ├── auth.py             # Kiro authentication
│   │   ├── http_client.py      # HTTP client with retry
│   │   ├── converters.py       # OpenAI -> Kiro conversion
│   │   ├── streaming.py        # Streaming response handler
│   │   ├── parsers.py          # AWS event stream parser
│   │   └── cache.py            # Model cache
│   └── utils/
│       └── helpers.py          # Utility functions
├── requirements.txt
├── .env.example
└── README.md
```

## Requirements

- Python 3.10+
- pip

## Installation

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

Edit `.env` file with your settings:

```env
# Required
PROXY_API_KEY=your_secure_api_key

# Kiro Credentials (optional for multi-tenant only mode)
REFRESH_TOKEN=your_kiro_refresh_token
PROFILE_ARN=your_profile_arn
KIRO_REGION=us-east-1
```

## Running the Application

### Development Mode

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or run directly:

```bash
python -m app.main
```

## API Endpoints

### Health Check
- `GET /` - API info
- `GET /health` - Detailed health status

### OpenAI Compatible
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions (streaming & non-streaming)

## Usage Examples

### List Models

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY"
```

### Chat Completion

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Streaming Chat

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

### Multi-tenant Mode

Pass refresh token in Authorization header:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY:YOUR_REFRESH_TOKEN" \
  -d '{"model": "claude-sonnet-4-5", "messages": [...]}'
```

## Available Models

- `claude-opus-4-5` / `claude-opus-4-5-20251101`
- `claude-sonnet-4-5` / `claude-sonnet-4-5-20250929`
- `claude-sonnet-4` / `claude-sonnet-4-20250514`
- `claude-haiku-4-5` / `claude-haiku-4-5-20251001`
- `claude-3-7-sonnet-20250219`

## Documentation

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## License

Based on KiroGate by Jwadow - AGPL-3.0 License
