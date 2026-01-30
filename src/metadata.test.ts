import { describe, it, expect } from 'vitest';
import { parseSafetensorsMetadata } from './metadata';

describe('metadata', () => {
  describe('parseSafetensorsMetadata', () => {
    it('should parse simple metadata with one component', () => {
      const rawMetadata = {
        Transformer: {
          'layer.0.weight': { dtype: 'F32', shape: [768, 768] },
          'layer.1.weight': { dtype: 'F32', shape: [768, 3072] },
        },
      };

      const result = parseSafetensorsMetadata(rawMetadata);

      expect(result.components).toHaveProperty('Transformer');
      expect(result.components.Transformer.paramCount).toBe(768 * 768 + 768 * 3072);
      expect(result.components.Transformer.bytesCount).toBe((768 * 768 + 768 * 3072) * 4);
      expect(result.paramCount).toBe(768 * 768 + 768 * 3072);
      expect(result.bytesCount).toBe((768 * 768 + 768 * 3072) * 4);
    });

    it('should parse metadata with multiple components', () => {
      const rawMetadata = {
        Transformer: {
          'layer.0.weight': { dtype: 'F32', shape: [100, 100] },
        },
        Dense: {
          'dense.weight': { dtype: 'F16', shape: [50, 100] },
        },
      };

      const result = parseSafetensorsMetadata(rawMetadata);

      expect(Object.keys(result.components)).toHaveLength(2);
      expect(result.components.Transformer.paramCount).toBe(100 * 100);
      expect(result.components.Transformer.bytesCount).toBe(100 * 100 * 4);
      expect(result.components.Dense.paramCount).toBe(50 * 100);
      expect(result.components.Dense.bytesCount).toBe(50 * 100 * 2);
      expect(result.paramCount).toBe(100 * 100 + 50 * 100);
      expect(result.bytesCount).toBe(100 * 100 * 4 + 50 * 100 * 2);
    });

    it('should handle different dtypes correctly', () => {
      const rawMetadata = {
        Transformer: {
          'weight1': { dtype: 'F64', shape: [10, 10] },
          'weight2': { dtype: 'F32', shape: [10, 10] },
          'weight3': { dtype: 'F16', shape: [10, 10] },
          'weight4': { dtype: 'I8', shape: [10, 10] },
        },
      };

      const result = parseSafetensorsMetadata(rawMetadata);

      expect(result.components.Transformer.dtypes.F64.paramCount).toBe(100);
      expect(result.components.Transformer.dtypes.F64.bytesCount).toBe(100 * 8);
      expect(result.components.Transformer.dtypes.F32.paramCount).toBe(100);
      expect(result.components.Transformer.dtypes.F32.bytesCount).toBe(100 * 4);
      expect(result.components.Transformer.dtypes.F16.paramCount).toBe(100);
      expect(result.components.Transformer.dtypes.F16.bytesCount).toBe(100 * 2);
      expect(result.components.Transformer.dtypes.I8.paramCount).toBe(100);
      expect(result.components.Transformer.dtypes.I8.bytesCount).toBe(100 * 1);
    });

    it('should ignore __metadata__ keys', () => {
      const rawMetadata = {
        Transformer: {
          '__metadata__': { some: 'metadata' },
          'weight': { dtype: 'F32', shape: [10, 10] },
        },
      };

      const result = parseSafetensorsMetadata(rawMetadata);

      expect(result.components.Transformer.paramCount).toBe(100);
      expect(result.components.Transformer.bytesCount).toBe(100 * 4);
    });

    it('should handle empty metadata', () => {
      const rawMetadata = {};

      const result = parseSafetensorsMetadata(rawMetadata);

      expect(result.components).toEqual({});
      expect(result.paramCount).toBe(0);
      expect(result.bytesCount).toBe(0);
    });

    it('should handle complex shapes', () => {
      const rawMetadata = {
        Transformer: {
          'weight': { dtype: 'F32', shape: [2, 3, 4, 5] },
        },
      };

      const result = parseSafetensorsMetadata(rawMetadata);

      const expectedParams = 2 * 3 * 4 * 5;
      expect(result.components.Transformer.paramCount).toBe(expectedParams);
      expect(result.components.Transformer.bytesCount).toBe(expectedParams * 4);
    });

    it('should handle quantized 4-bit dtypes (INT4, NF4, FP4)', () => {
      const rawMetadata = {
        Transformer: {
          'weight.q': { dtype: 'INT4', shape: [1000, 1000] },
          'weight.scale': { dtype: 'F32', shape: [1000] },
        },
      };

      const result = parseSafetensorsMetadata(rawMetadata);

      const int4Params = 1000 * 1000;
      const f32Params = 1000;
      expect(result.components.Transformer.dtypes.INT4.paramCount).toBe(int4Params);
      expect(result.components.Transformer.dtypes.INT4.bytesCount).toBe(int4Params * 0.5);
      expect(result.components.Transformer.dtypes.F32.paramCount).toBe(f32Params);
      expect(result.components.Transformer.dtypes.F32.bytesCount).toBe(f32Params * 4);
      expect(result.paramCount).toBe(int4Params + f32Params);
      expect(result.bytesCount).toBe(int4Params * 0.5 + f32Params * 4);
    });
  });
});
