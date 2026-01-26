import { describe, it, expect } from 'vitest';
import { parseMetadataSize, combineArrayBuffers } from './binary';

describe('binary', () => {
  describe('parseMetadataSize', () => {
    it('should parse a small metadata size correctly', () => {
      // Create a buffer with metadata size of 100 (0x64 in hex)
      const buffer = new ArrayBuffer(8);
      const view = new DataView(buffer);
      view.setUint32(0, 100, true); // little-endian
      view.setUint32(4, 0, true);

      expect(parseMetadataSize(buffer)).toBe(100);
    });

    it('should parse a larger metadata size correctly', () => {
      // Create a buffer with metadata size of 50000
      const buffer = new ArrayBuffer(8);
      const view = new DataView(buffer);
      view.setUint32(0, 50000, true);
      view.setUint32(4, 0, true);

      expect(parseMetadataSize(buffer)).toBe(50000);
    });

    it('should handle large metadata sizes with high bits set', () => {
      // Create a buffer with metadata size that uses high 32 bits
      const buffer = new ArrayBuffer(8);
      const view = new DataView(buffer);
      view.setUint32(0, 0xFFFFFFFF, true);
      view.setUint32(4, 1, true);

      expect(parseMetadataSize(buffer)).toBe(0xFFFFFFFF + 0x100000000);
    });
  });

  describe('combineArrayBuffers', () => {
    it('should combine two ArrayBuffers', () => {
      const buffer1 = new Uint8Array([1, 2, 3]).buffer;
      const buffer2 = new Uint8Array([4, 5, 6]).buffer;

      const combined = combineArrayBuffers(buffer1, buffer2);
      const view = new Uint8Array(combined);

      expect(view.length).toBe(6);
      expect(Array.from(view)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('should handle empty buffers', () => {
      const buffer1 = new Uint8Array([]).buffer;
      const buffer2 = new Uint8Array([1, 2]).buffer;

      const combined = combineArrayBuffers(buffer1, buffer2);
      const view = new Uint8Array(combined);

      expect(view.length).toBe(2);
      expect(Array.from(view)).toEqual([1, 2]);
    });

    it('should handle both buffers being empty', () => {
      const buffer1 = new Uint8Array([]).buffer;
      const buffer2 = new Uint8Array([]).buffer;

      const combined = combineArrayBuffers(buffer1, buffer2);
      const view = new Uint8Array(combined);

      expect(view.length).toBe(0);
    });
  });
});
