# API

## Endpoints

- `GET /health`
- `GET /models`
- `POST /generate`
- `POST /chat`
- `POST /explain`
- `POST /fix`
- `POST /refactor`
- `POST /tests`
- `POST /edit`

## Authentication

If `LLM_API_KEY` is configured, send it as `X-API-Key`.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/fix \
  -H "Content-Type: application/json" \
  -d '{
    "selection": "def total(values): return 0",
    "language": "python",
    "parameters": {
      "temperature": 0.2,
      "max_new_tokens": 256,
      "top_p": 0.95,
      "do_sample": true,
      "response_format": "text"
    }
  }'
```
