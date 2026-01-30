import { describe, it, expect } from 'vitest';
import { getSafetensorsDtypeBytes, RuntimeError } from './types';

describe('types', () => {
  describe('getSafetensorsDtypeBytes', () => {
    it('should return 8 for 64-bit types', () => {
      expect(getSafetensorsDtypeBytes('F64')).toBe(8);
      expect(getSafetensorsDtypeBytes('I64')).toBe(8);
      expect(getSafetensorsDtypeBytes('U64')).toBe(8);
    });

    it('should return 4 for 32-bit types', () => {
      expect(getSafetensorsDtypeBytes('F32')).toBe(4);
      expect(getSafetensorsDtypeBytes('I32')).toBe(4);
      expect(getSafetensorsDtypeBytes('U32')).toBe(4);
    });

    it('should return 2 for 16-bit types', () => {
      expect(getSafetensorsDtypeBytes('F16')).toBe(2);
      expect(getSafetensorsDtypeBytes('BF16')).toBe(2);
      expect(getSafetensorsDtypeBytes('I16')).toBe(2);
      expect(getSafetensorsDtypeBytes('U16')).toBe(2);
    });

    it('should return 1 for 8-bit types', () => {
      expect(getSafetensorsDtypeBytes('F8_E5M2')).toBe(1);
      expect(getSafetensorsDtypeBytes('F8_E4M3')).toBe(1);
      expect(getSafetensorsDtypeBytes('I8')).toBe(1);
      expect(getSafetensorsDtypeBytes('U8')).toBe(1);
    });

    it('should return 0.5 for 4-bit quantized types', () => {
      expect(getSafetensorsDtypeBytes('INT4')).toBe(0.5);
      expect(getSafetensorsDtypeBytes('NF4')).toBe(0.5);
      expect(getSafetensorsDtypeBytes('FP4')).toBe(0.5);
      expect(getSafetensorsDtypeBytes('FP4_E2M1')).toBe(0.5);
    });

    it('should throw RuntimeError for unknown dtype', () => {
      expect(() => getSafetensorsDtypeBytes('UNKNOWN')).toThrow(RuntimeError);
      expect(() => getSafetensorsDtypeBytes('UNKNOWN')).toThrow('DTYPE=UNKNOWN NOT HANDLED');
    });
  });

  describe('RuntimeError', () => {
    it('should create an error with the correct name', () => {
      const error = new RuntimeError('Test error');
      expect(error.name).toBe('RuntimeError');
      expect(error.message).toBe('Test error');
      expect(error).toBeInstanceOf(Error);
    });
  });
});
