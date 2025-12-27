# FrugalAI - OpenRouter LLM Proxy

An intelligent LLM proxy that automatically routes requests to the best free model available on OpenRouter. Provides both OpenAI and Anthropic compatible API endpoints.

## Features

- **Automatic Model Selection**: Intelligently selects the best free model based on:
  - Parameter count (model size)
  - Popularity score
  - Context length
  - Architecture preferences
- **Dual API Compatibility**: Works with both OpenAI and Anthropic client libraries
- **Smart Caching**: Caches model list to reduce API calls
- **Configurable Constraints**: Set minimum parameter counts and popularity thresholds
- **Streaming Support**: Full support for streaming responses

## Installation

```bash
go install github.com/mosajjal/frugalai/cmd/frugalai@latest
```

Or build from source:

```bash
git clone https://github.com/mosajjal/frugalai.git
cd frugalai
go build -o frugalai ./cmd/frugalai
```

## Usage

### Basic Usage

```bash
# Using environment variables
export FRUGALAI_API_KEY="your-openrouter-api-key"
frugalai

# Using CLI flags
frugalai -api-key "your-openrouter-api-key"
```

### Configuration Options

| Flag | Environment Variable | Default | Description |
|------|---------------------|---------|-------------|
| `-api-key`, `-k` | `FRUGALAI_API_KEY` | *required* | OpenRouter API key |
| `-port`, `-p` | `FRUGALAI_PORT` | `8080` | Server port |
| `-min-params` | `FRUGALAI_MIN_PARAMS` | `0` | Minimum parameter count |
| `-min-popularity` | `FRUGALAI_MIN_POPULARITY` | `0` | Minimum popularity score |
| `-enable-openai` | - | `true` | Enable OpenAI-compatible API |
| `-enable-anthropic` | - | `true` | Enable Anthropic-compatible API |
| `-openai-path` | - | `/v1` | OpenAI endpoint path |
| `-anthropic-path` | - | `/v1` | Anthropic endpoint path |
| `-log-level` | `FRUGALAI_LOG_LEVEL` | `info` | Log level |
| `-cache-ttl` | `FRUGALAI_CACHE_TTL` | `300` | Model cache TTL (seconds) |
| `-preferred-arch` | `FRUGALAI_PREFERRED_ARCH` | - | Preferred architectures (comma-separated) |

### Example Configurations

**Only use models with at least 30B parameters:**
```bash
frugalai -k "$API_KEY" -min-params 30000000000
```

**Prefer transformer-based models:**
```bash
frugalai -k "$API_KEY" -preferred-arch "transformer,llama"
```

**Run on custom port with debug logging:**
```bash
frugalai -k "$API_KEY" -p 9000 -log-level debug
```

## API Endpoints

### OpenAI-Compatible API

```
POST http://localhost:8080/v1/chat/completions
GET  http://localhost:8080/v1/models
```

### Anthropic-Compatible API

```
POST http://localhost:8080/v1/messages
```

### Utility Endpoints

```
GET http://localhost:8080/health     # Health check
GET http://localhost:8080/model      # Current selected model info
```

## Client Examples

### OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="any-key"  # Not used by proxy
)

response = client.chat.completions.create(
    model="auto",  # Let proxy select best model
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Anthropic Python Client

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080/v1",
    api_key="any-key"  # Not used by proxy
)

message = client.messages.create(
    model="claude-3-haiku",  # Will be auto-selected
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(message.content[0].text)
```

### cURL Examples

```bash
# OpenAI format
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Anthropic format
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-haiku",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Model Selection Algorithm

The proxy scores models based on several factors:

1. **Popularity** (30% weight): Logarithmic scale based on usage
2. **Parameters** (40% weight): Larger models get higher scores
3. **Context Length** (20% weight): Longer context is preferred
4. **Architecture Bonus** (10%): Bonus for preferred architectures
5. **Quality Bonus**: Additional bonus for known high-quality model families

Quality bonuses are applied for:
- Claude/Anthropic models: +0.15
- GPT/OpenAI models: +0.12
- Gemini/Google models: +0.10
- Mistral/Mixtral: +0.08
- Llama/Meta: +0.08

## Getting an OpenRouter API Key

1. Visit [OpenRouter.ai](https://openrouter.ai)
2. Sign up for a free account
3. Get your API key from the settings page

Free models on OpenRouter rotate periodically. This proxy automatically selects the best available free model at any given time.

## Development

```bash
# Run tests
go test ./...

# Run with coverage
go test -cover ./...

# Build
go build -o frugalai ./cmd/frugalai
```

## License

MIT License
