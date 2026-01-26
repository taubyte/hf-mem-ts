import { describe, it, expect, vi, beforeEach } from 'vitest';
import { run } from './api';

// Mock fetch globally
global.fetch = vi.fn();

describe('api', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('run', () => {
    it('should handle authentication errors correctly', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
      });

      await expect(
        run('test/model', 'main', { token: 'invalid_token' })
      ).rejects.toThrow('Authentication failed (401)');
    });

    it('should handle missing model files error', async () => {
      // Mock file tree response with no safetensors files
      (global.fetch as any)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => [
            { path: 'config.json', type: 'file' },
            { path: 'tokenizer.json', type: 'file' },
          ],
        });

      await expect(run('test/model')).rejects.toThrow(
        'NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json` HAS BEEN FOUND'
      );
    });

    it('should process single model.safetensors file', async () => {
      // Mock file tree
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { path: 'model.safetensors', type: 'file' },
          { path: 'config.json', type: 'file' },
        ],
      });

      // Mock safetensors metadata fetch
      const mockMetadata = {
        'layer.0.weight': { dtype: 'F32', shape: [100, 100] },
      };
      const mockMetadataJson = JSON.stringify(mockMetadata);
      const metadataSize = mockMetadataJson.length;

      // Create mock safetensors binary format: 8 bytes (size) + metadata
      const buffer = new ArrayBuffer(8 + metadataSize);
      const view = new DataView(buffer);
      view.setUint32(0, metadataSize, true);
      view.setUint32(4, 0, true);
      const textEncoder = new TextEncoder();
      const metadataBytes = textEncoder.encode(mockMetadataJson);
      new Uint8Array(buffer, 8).set(metadataBytes);

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        arrayBuffer: async () => buffer,
      });

      const result = await run('test/model');

      expect(result.metadata).toBeDefined();
      expect(result.metadata.components).toHaveProperty('Transformer');
      expect(result.metadata.paramCount).toBe(100 * 100);
    });

    it('should include json output when requested', async () => {
      // Mock file tree
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { path: 'model.safetensors', type: 'file' },
        ],
      });

      // Mock safetensors metadata
      const mockMetadata = {
        'layer.0.weight': { dtype: 'F32', shape: [10, 10] },
      };
      const mockMetadataJson = JSON.stringify(mockMetadata);
      const metadataSize = mockMetadataJson.length;

      const buffer = new ArrayBuffer(8 + metadataSize);
      const view = new DataView(buffer);
      view.setUint32(0, metadataSize, true);
      view.setUint32(4, 0, true);
      const textEncoder = new TextEncoder();
      const metadataBytes = textEncoder.encode(mockMetadataJson);
      new Uint8Array(buffer, 8).set(metadataBytes);

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        arrayBuffer: async () => buffer,
      });

      const result = await run('test/model', 'main', { jsonOutput: true });

      expect(result.json).toBeDefined();
      expect(result.json.model_id).toBe('test/model');
      expect(result.json.revision).toBe('main');
      expect(result.json.param_count).toBe(100);
    });

    it('should use token when provided', async () => {
      // Mock file tree
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { path: 'model.safetensors', type: 'file' },
        ],
      });

      // Mock safetensors metadata
      const mockMetadata = {
        'layer.0.weight': { dtype: 'F32', shape: [10, 10] },
      };
      const mockMetadataJson = JSON.stringify(mockMetadata);
      const metadataSize = mockMetadataJson.length;

      const buffer = new ArrayBuffer(8 + metadataSize);
      const view = new DataView(buffer);
      view.setUint32(0, metadataSize, true);
      view.setUint32(4, 0, true);
      const textEncoder = new TextEncoder();
      const metadataBytes = textEncoder.encode(mockMetadataJson);
      new Uint8Array(buffer, 8).set(metadataBytes);

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        arrayBuffer: async () => buffer,
      });

      await run('test/model', 'main', { token: 'hf_test_token' });

      // Check that fetch was called with Authorization header
      const fetchCalls = (global.fetch as any).mock.calls;
      expect(fetchCalls.length).toBeGreaterThan(0);
      const firstCall = fetchCalls[0];
      expect(firstCall[1].headers.Authorization).toBe('Bearer hf_test_token');
    });
  });
});
