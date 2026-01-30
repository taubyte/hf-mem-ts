import { describe, it, expect } from 'vitest';
import { run } from './api';

describe('api integration', () => {
  it(
    'should handle MiniMaxAI/MiniMax-M2 (sharded model)',
    async () => {
      const result = await run('MiniMaxAI/MiniMax-M2');
      expect(result.metadata).toBeDefined();
      expect(result.metadata.paramCount).toBeGreaterThan(0);
      expect(result.metadata.bytesCount).toBeGreaterThan(0);
      expect(result.metadata.components).toHaveProperty('Transformer');
    },
    90_000
  );
});
