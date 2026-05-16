# Vulpes FrugalAI Plugin

This binary turns FrugalAI into a Vulpes `upstream_provider` plugin without moving the project into the Vulpes plugin repository.

It keeps FrugalAI's core behavior:

- fetches OpenRouter models
- ranks free models with the existing selector
- probes candidates before promotion
- exposes an `auto` provider model for Vulpes aliases
- fails over to the next working candidate on retryable upstream errors
- reports usage with the actual OpenRouter model as `provider_model`

## Build

```bash
go build -o bin/vulpes-frugalai-plugin ./cmd/vulpes-frugalai-plugin
```

The module currently uses the sibling Vulpes SDK checkout:

```text
replace github.com/FenkoHQ/vulpes-core-plugins/sdk => ../vulpes-core-plugins/sdk
```

## Vulpes config example

```yaml
plugins:
  - name: frugalai
    source:
      type: filesystem
      path: /opt/frugalai/bin/vulpes-frugalai-plugin
    capabilities: [upstream_provider]
    fail_mode: closed
    config:
      api_key: ${secret:OPENROUTER_API_KEY}
      min_params: 30000000000
      num_candidates: 10
      probe_on_start: true
      max_retries: 3

pipeline:
  upstream_providers: [frugalai]

models:
  aliases:
    frugal:
      candidates:
        - provider: frugalai
          model: auto
          weight: 100
```

Use route property `force_model` if a Vulpes route should pin a specific OpenRouter model while still using this plugin:

```yaml
properties:
  force_model: qwen/qwen3-next-80b-a3b-instruct:free
```
