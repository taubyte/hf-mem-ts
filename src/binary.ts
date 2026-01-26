export function parseMetadataSize(buffer: ArrayBuffer): number {
  const view = new DataView(buffer);
  const low = view.getUint32(0, true);
  const high = view.getUint32(4, true);
  return low + high * 0x100000000;
}

export function extractMetadata(buffer: ArrayBuffer, metadataSize: number, maxMetadataSize: number): string {
  if (metadataSize < maxMetadataSize) {
    const metadataBytes = new Uint8Array(buffer, 8, metadataSize);
    return new TextDecoder().decode(metadataBytes);
  } else {
    const metadataBytes = new Uint8Array(buffer, 8, maxMetadataSize);
    return new TextDecoder().decode(metadataBytes);
  }
}

export function combineArrayBuffers(buffer1: ArrayBuffer, buffer2: ArrayBuffer): ArrayBuffer {
  const combined = new Uint8Array(buffer1.byteLength + buffer2.byteLength);
  combined.set(new Uint8Array(buffer1), 0);
  combined.set(new Uint8Array(buffer2), buffer1.byteLength);
  return combined.buffer;
}
