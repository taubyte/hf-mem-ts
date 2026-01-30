import { parseMetadataSize } from "./binary.js";
import { parseSafetensorsMetadata, SafetensorsMetadata } from "./metadata.js";

const MAX_METADATA_SIZE = 200_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 60_000;

function handleHttpError(response: Response, headers: Record<string, string>): never {
  if (response.status === 401) {
    const hasToken = !!headers.Authorization;
    const errorMsg = hasToken
      ? "Authentication failed (401). The provided token may be invalid or expired. Please check your token. Also verify the model ID is correct (case-sensitive)."
      : "Authentication failed (401). This model may be private or require authentication. Please provide a Hugging Face token. Also verify the model ID is correct (case-sensitive).";
    throw new Error(errorMsg);
  }
  throw new Error(`HTTP error! status: ${response.status}${response.statusText ? ` - ${response.statusText}` : ''}`);
}

function isAbortError(e: unknown): boolean {
  return e instanceof Error && e.name === "AbortError";
}

async function getJsonFile(
  url: string,
  headers: Record<string, string> = {},
  timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS
): Promise<any> {
  const attempt = async (): Promise<any> => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { headers, signal: controller.signal });
      if (!response.ok) {
        handleHttpError(response, headers);
      }
      return response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  };
  try {
    return await attempt();
  } catch (e) {
    if (isAbortError(e)) {
      return await attempt();
    }
    throw e;
  }
}

async function fetchSafetensorsMetadata(
  url: string,
  headers: Record<string, string> = {},
  timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS
): Promise<Record<string, any>> {
  const attempt = async (): Promise<Record<string, any>> => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        headers: { Range: `bytes=0-${MAX_METADATA_SIZE}`, ...headers },
        signal: controller.signal,
      });

      if (!response.ok) {
        handleHttpError(response, headers);
      }

      const buffer = await response.arrayBuffer();
      const metadataSize = parseMetadataSize(buffer);

      if (metadataSize > MAX_METADATA_SIZE) {
        const remainderController = new AbortController();
        const remainderTimeoutId = setTimeout(() => remainderController.abort(), timeoutMs);
        try {
          const remainderResponse = await fetch(url, {
            headers: {
              Range: `bytes=${MAX_METADATA_SIZE + 1}-${metadataSize + 7}`,
              ...headers,
            },
            signal: remainderController.signal,
          });
          if (!remainderResponse.ok) {
            throw new Error(`HTTP error! status: ${remainderResponse.status}`);
          }
          const remainderBuffer = await remainderResponse.arrayBuffer();
          const combined = new Uint8Array(buffer.byteLength + remainderBuffer.byteLength);
          combined.set(new Uint8Array(buffer), 0);
          combined.set(new Uint8Array(remainderBuffer), buffer.byteLength);
          const metadataBytes = new Uint8Array(combined.buffer, 8, metadataSize);
          const metadataText = new TextDecoder().decode(metadataBytes);
          return JSON.parse(metadataText);
        } finally {
          clearTimeout(remainderTimeoutId);
        }
      }

      const metadataBytes = new Uint8Array(buffer, 8, metadataSize);
      const metadataText = new TextDecoder().decode(metadataBytes);
      return JSON.parse(metadataText);
    } finally {
      clearTimeout(timeoutId);
    }
  };
  try {
    return await attempt();
  } catch (e) {
    if (isAbortError(e)) {
      return await attempt();
    }
    throw e;
  }
}

async function fetchModulesAndDenseMetadata(
  baseUrl: string,
  headers: Record<string, string> = {},
  timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS
): Promise<Record<string, Record<string, any>>> {
  const modules = await getJsonFile(`${baseUrl}/modules.json`, headers, timeoutMs);
  const paths = modules
    .filter((m: any) => m.type === "sentence_transformers.models.Dense" && m.path)
    .map((m: any) => m.path);

  const result: Record<string, Record<string, any>> = {};
  for (const path of paths) {
    result[path] = await fetchSafetensorsMetadata(
      `${baseUrl}/${path}/model.safetensors`,
      headers,
      timeoutMs
    );
  }
  return result;
}

async function buildMetadata(
  metadata: Record<string, any>,
  filePaths: string[],
  baseUrl: string,
  headers: Record<string, string>,
  timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS
): Promise<Record<string, Record<string, any>>> {
  const isSentenceTransformer = filePaths.includes("config_sentence_transformers.json");
  if (isSentenceTransformer) {
    const denseMetadata = filePaths.includes("modules.json")
      ? await fetchModulesAndDenseMetadata(baseUrl, headers, timeoutMs)
      : {};
    return { "0_Transformer": metadata, ...denseMetadata };
  }
  return { Transformer: metadata };
}

async function handleSingleFile(
  baseUrl: string,
  filePaths: string[],
  headers: Record<string, string>,
  timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS
): Promise<Record<string, Record<string, any>>> {
  const metadata = await fetchSafetensorsMetadata(`${baseUrl}/model.safetensors`, headers, timeoutMs);
  return buildMetadata(metadata, filePaths, baseUrl, headers, timeoutMs);
}

async function handleShardedFile(
  baseUrl: string,
  filePaths: string[],
  headers: Record<string, string>,
  timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS
): Promise<Record<string, Record<string, any>>> {
  const filesIndex = await getJsonFile(`${baseUrl}/model.safetensors.index.json`, headers, timeoutMs);
  const urls = Array.from(
    new Set(Object.values(filesIndex.weight_map).map((f: any) => `${baseUrl}/${f}`))
  );

  const mergedMetadata: Record<string, any> = {};
  for (const url of urls) {
    const metadata = await fetchSafetensorsMetadata(url, headers, timeoutMs);
    Object.assign(mergedMetadata, metadata);
  }

  return buildMetadata(mergedMetadata, filePaths, baseUrl, headers, timeoutMs);
}

function findDiffusersUrls(
  path: string,
  filePaths: string[],
  baseUrl: string,
  headers: Record<string, string>,
  timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS
): Promise<string[]> {
  if (filePaths.includes(`${path}/diffusion_pytorch_model.safetensors`)) {
    return Promise.resolve([`${baseUrl}/${path}/diffusion_pytorch_model.safetensors`]);
  }
  if (filePaths.includes(`${path}/model.safetensors`)) {
    return Promise.resolve([`${baseUrl}/${path}/model.safetensors`]);
  }
  if (filePaths.includes(`${path}/diffusion_pytorch_model.safetensors.index.json`)) {
    return getJsonFile(`${baseUrl}/${path}/diffusion_pytorch_model.safetensors.index.json`, headers, timeoutMs).then(
      (index) => Object.values(index.weight_map).map((f: any) => `${baseUrl}/${path}/${f}`)
    );
  }
  if (filePaths.includes(`${path}/model.safetensors.index.json`)) {
    return getJsonFile(`${baseUrl}/${path}/model.safetensors.index.json`, headers, timeoutMs).then(
      (index) => Object.values(index.weight_map).map((f: any) => `${baseUrl}/${path}/${f}`)
    );
  }
  return Promise.resolve([]);
}

async function handleDiffusersModel(
  baseUrl: string,
  filePaths: string[],
  headers: Record<string, string>,
  timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS
): Promise<Record<string, Record<string, any>>> {
  const filesIndex = await getJsonFile(`${baseUrl}/model_index.json`, headers, timeoutMs);
  const paths = Object.keys(filesIndex).filter((k) => !k.startsWith("_"));

  const pathUrls: Record<string, string[]> = {};
  for (const path of paths) {
    const urls = await findDiffusersUrls(path, filePaths, baseUrl, headers, timeoutMs);
    if (urls.length > 0) {
      pathUrls[path] = urls;
    }
  }

  const result: Record<string, Record<string, any>> = {};
  for (const [path, pathUrlList] of Object.entries(pathUrls)) {
    const pathMetadata: Record<string, any> = {};
    for (const url of pathUrlList) {
      const metadata = await fetchSafetensorsMetadata(url, headers, timeoutMs);
      Object.assign(pathMetadata, metadata);
    }
    result[path] = pathMetadata;
  }

  return result;
}

export async function run(
  modelId: string,
  revision: string = "main",
  options?: {
    token?: string;
    jsonOutput?: boolean;
    /** Request timeout in milliseconds. Default: 60000 (60s). */
    timeoutMs?: number;
  }
): Promise<{ metadata: SafetensorsMetadata; json?: any }> {
  const headers: Record<string, string> = {
    "User-Agent": `hf-mem/0.3; id=${Date.now()}; model_id=${modelId}; revision=${revision}`,
    ...(options?.token ? { Authorization: `Bearer ${options.token}` } : {}),
  };

  const timeoutMs = options?.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;

  const treeUrl = `https://huggingface.co/api/models/${modelId}/tree/${revision}?recursive=true`;
  const files = await getJsonFile(treeUrl, headers, timeoutMs);
  const filePaths = files.filter((f: any) => f.path && f.type === "file").map((f: any) => f.path);
  const baseUrl = `https://huggingface.co/${modelId}/resolve/${revision}`;

  let rawMetadata: Record<string, Record<string, any>>;

  if (filePaths.includes("model.safetensors")) {
    rawMetadata = await handleSingleFile(baseUrl, filePaths, headers, timeoutMs);
  } else if (filePaths.includes("model.safetensors.index.json")) {
    rawMetadata = await handleShardedFile(baseUrl, filePaths, headers, timeoutMs);
  } else if (filePaths.includes("model_index.json")) {
    rawMetadata = await handleDiffusersModel(baseUrl, filePaths, headers, timeoutMs);
  } else {
    throw new Error(
      "NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json` HAS BEEN FOUND"
    );
  }

  const metadata = parseSafetensorsMetadata(rawMetadata);

  if (options?.jsonOutput) {
    return {
      metadata,
      json: {
        model_id: modelId,
        revision,
        components: metadata.components,
        param_count: metadata.paramCount,
        bytes_count: metadata.bytesCount,
      },
    };
  }

  return { metadata };
}
