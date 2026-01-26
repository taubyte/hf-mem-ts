# hf-mem

Estimate inference memory requirements for Hugging Face models.

## Installation

```bash
npm install hf-mem
```

## Usage

```typescript
import { run } from 'hf-mem';

// Basic usage
const result = await run('MiniMaxAI/MiniMax-M2');

console.log(result.metadata.bytesCount); // Total bytes
console.log(result.metadata.paramCount); // Total parameters

// With options
const resultWithOptions = await run('MiniMaxAI/MiniMax-M2', 'main', {
  token: 'hf_...', // Optional: for private models
  jsonOutput: true // Optional: include JSON output
});

if (resultWithOptions.json) {
  console.log(resultWithOptions.json);
}
```

## API

### `run(modelId, revision?, options?)`

Estimates memory requirements for a Hugging Face model.

**Parameters:**
- `modelId` (string): The Hugging Face model ID (e.g., `'MiniMaxAI/MiniMax-M2'`)
- `revision` (string, optional): Model revision (default: `"main"`)
- `options` (object, optional):
  - `token` (string, optional): Hugging Face token for private models
  - `jsonOutput` (boolean, optional): If `true`, includes JSON output in result

**Returns:**
- `Promise<{ metadata: SafetensorsMetadata; json?: any }>`

**Example:**
```typescript
const result = await run('bert-base-uncased', 'main', {
  token: process.env.HF_TOKEN,
  jsonOutput: false
});

// Access metadata
console.log(`Total parameters: ${result.metadata.paramCount}`);
console.log(`Total bytes: ${result.metadata.bytesCount}`);
console.log(`Components:`, Object.keys(result.metadata.components));
```

## Types

```typescript
import type {
  SafetensorsMetadata,
  ComponentMetadata,
  DtypeMetadata,
  SafetensorsDtypes
} from 'hf-mem';

import { RuntimeError } from 'hf-mem';
```

## Supported Model Types

- **Transformers models**: Standard PyTorch models with `model.safetensors` or sharded variants
- **Diffusers models**: Models with `model_index.json` and component-based structure
- **Sentence Transformers**: Models with additional Dense layers

## How It Works

The library:
1. Fetches the model's file tree from the Hugging Face Hub API
2. Uses HTTP Range requests to fetch only the metadata portion of safetensors files (first ~100KB)
3. Parses the binary safetensors format to extract tensor metadata
4. Calculates memory requirements based on tensor shapes and data types

## Browser and Node.js Support

This package works in both Node.js (18+) and modern browsers that support the `fetch` API.

## License

MIT
