import { parseMetadataSize } from "./binary.js";
import { parseSafetensorsMetadata, SafetensorsMetadata } from "./metadata.js";

const MAX_METADATA_SIZE = 200_000;
const REQUEST_TIMEOUT = 10_000;

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

async function getJsonFile(url: string, headers: Record<string, string> = {}): Promise<any> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

  try {
    const response = await fetch(url, { headers, signal: controller.signal });
    if (!response.ok) {
      handleHttpError(response, headers);
    }
    return response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

async function fetchSafetensorsMetadata(
  url: string,
  headers: Record<string, string> = {}
): Promise<Record<string, any>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

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
      const remainderResponse = await fetch(url, {
        headers: {
          Range: `bytes=${MAX_METADATA_SIZE + 1}-${metadataSize + 7}`,
          ...headers,
        },
        signal: controller.signal,
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
    }

    const metadataBytes = new Uint8Array(buffer, 8, metadataSize);
    const metadataText = new TextDecoder().decode(metadataBytes);
    return JSON.parse(metadataText);
  } finally {
    clearTimeout(timeoutId);
  }
}

async function fetchModulesAndDenseMetadata(
  baseUrl: string,
  headers: Record<string, string> = {}
): Promise<Record<string, Record<string, any>>> {
  const modules = await getJsonFile(`${baseUrl}/modules.json`, headers);
  const paths = modules
    .filter((m: any) => m.type === "sentence_transformers.models.Dense" && m.path)
    .map((m: any) => m.path);

  const metadataPromises = paths.map((path: string) =>
    fetchSafetensorsMetadata(`${baseUrl}/${path}/model.safetensors`, headers)
  );

  const metadataList = await Promise.all(metadataPromises);
  const result: Record<string, Record<string, any>> = {};
  paths.forEach((path: string, i: number) => {
    result[path] = metadataList[i];
  });
  return result;
}

async function buildMetadata(
  metadata: Record<string, any>,
  filePaths: string[],
  baseUrl: string,
  headers: Record<string, string>
): Promise<Record<string, Record<string, any>>> {
  const isSentenceTransformer = filePaths.includes("config_sentence_transformers.json");
  if (isSentenceTransformer) {
    const denseMetadata = filePaths.includes("modules.json")
      ? await fetchModulesAndDenseMetadata(baseUrl, headers)
      : {};
    return { "0_Transformer": metadata, ...denseMetadata };
  }
  return { Transformer: metadata };
}

async function handleSingleFile(
  baseUrl: string,
  filePaths: string[],
  headers: Record<string, string>
): Promise<Record<string, Record<string, any>>> {
  const metadata = await fetchSafetensorsMetadata(`${baseUrl}/model.safetensors`, headers);
  return buildMetadata(metadata, filePaths, baseUrl, headers);
}

async function handleShardedFile(
  baseUrl: string,
  filePaths: string[],
  headers: Record<string, string>
): Promise<Record<string, Record<string, any>>> {
  const filesIndex = await getJsonFile(`${baseUrl}/model.safetensors.index.json`, headers);
  const urls = Array.from(
    new Set(Object.values(filesIndex.weight_map).map((f: any) => `${baseUrl}/${f}`))
  );

  const metadataList = await Promise.all(
    urls.map((url) => fetchSafetensorsMetadata(url, headers))
  );

  const mergedMetadata = metadataList.reduce(
    (acc, metadata) => ({ ...acc, ...metadata }),
    {}
  );

  return buildMetadata(mergedMetadata, filePaths, baseUrl, headers);
}

function findDiffusersUrls(
  path: string,
  filePaths: string[],
  baseUrl: string,
  headers: Record<string, string>
): Promise<string[]> {
  if (filePaths.includes(`${path}/diffusion_pytorch_model.safetensors`)) {
    return Promise.resolve([`${baseUrl}/${path}/diffusion_pytorch_model.safetensors`]);
  }
  if (filePaths.includes(`${path}/model.safetensors`)) {
    return Promise.resolve([`${baseUrl}/${path}/model.safetensors`]);
  }
  if (filePaths.includes(`${path}/diffusion_pytorch_model.safetensors.index.json`)) {
    return getJsonFile(`${baseUrl}/${path}/diffusion_pytorch_model.safetensors.index.json`, headers).then(
      (index) => Object.values(index.weight_map).map((f: any) => `${baseUrl}/${path}/${f}`)
    );
  }
  if (filePaths.includes(`${path}/model.safetensors.index.json`)) {
    return getJsonFile(`${baseUrl}/${path}/model.safetensors.index.json`, headers).then(
      (index) => Object.values(index.weight_map).map((f: any) => `${baseUrl}/${path}/${f}`)
    );
  }
  return Promise.resolve([]);
}

async function handleDiffusersModel(
  baseUrl: string,
  filePaths: string[],
  headers: Record<string, string>
): Promise<Record<string, Record<string, any>>> {
  const filesIndex = await getJsonFile(`${baseUrl}/model_index.json`, headers);
  const paths = Object.keys(filesIndex).filter((k) => !k.startsWith("_"));

  const pathUrlsEntries = await Promise.all(
    paths.map(async (path) => {
      const urls = await findDiffusersUrls(path, filePaths, baseUrl, headers);
      return [path, urls] as const;
    })
  );

  const pathUrls = Object.fromEntries(pathUrlsEntries.filter(([, urls]) => urls.length > 0));

  const allUrls = Object.values(pathUrls).flat();
  const metadataList = await Promise.all(allUrls.map((url) => fetchSafetensorsMetadata(url, headers)));

  const result: Record<string, Record<string, any>> = {};
  let urlIndex = 0;
  for (const [path, urls] of Object.entries(pathUrls)) {
    const pathMetadata = metadataList.slice(urlIndex, urlIndex + urls.length);
    result[path] = pathMetadata.reduce((acc, metadata) => ({ ...acc, ...metadata }), {});
    urlIndex += urls.length;
  }

  return result;
}

export async function run(
  modelId: string,
  revision: string = "main",
  options?: {
    token?: string;
    jsonOutput?: boolean;
  }
): Promise<{ metadata: SafetensorsMetadata; json?: any }> {
  const headers: Record<string, string> = {
    "User-Agent": `hf-mem/0.3; id=${Date.now()}; model_id=${modelId}; revision=${revision}`,
    ...(options?.token ? { Authorization: `Bearer ${options.token}` } : {}),
  };

  const treeUrl = `https://huggingface.co/api/models/${modelId}/tree/${revision}?recursive=true`;
  const files = await getJsonFile(treeUrl, headers);
  const filePaths = files.filter((f: any) => f.path && f.type === "file").map((f: any) => f.path);
  const baseUrl = `https://huggingface.co/${modelId}/resolve/${revision}`;

  let rawMetadata: Record<string, Record<string, any>>;

  if (filePaths.includes("model.safetensors")) {
    rawMetadata = await handleSingleFile(baseUrl, filePaths, headers);
  } else if (filePaths.includes("model.safetensors.index.json")) {
    rawMetadata = await handleShardedFile(baseUrl, filePaths, headers);
  } else if (filePaths.includes("model_index.json")) {
    rawMetadata = await handleDiffusersModel(baseUrl, filePaths, headers);
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
