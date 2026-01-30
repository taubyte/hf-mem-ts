export type SafetensorsDtypes =
  | "F64"
  | "I64"
  | "U64"
  | "F32"
  | "I32"
  | "U32"
  | "F16"
  | "BF16"
  | "I16"
  | "U16"
  | "F8_E5M2"
  | "F8_E4M3"
  | "I8"
  | "U8"
  /** 4-bit quantized (INT4, NF4, FP4, etc.); 0.5 bytes per parameter when packed. */
  | "INT4"
  | "NF4"
  | "FP4"
  | "FP4_E2M1";

export function getSafetensorsDtypeBytes(dtype: string): number {
  switch (dtype) {
    case "F64":
    case "I64":
    case "U64":
      return 8;
    case "F32":
    case "I32":
    case "U32":
      return 4;
    case "F16":
    case "BF16":
    case "I16":
    case "U16":
      return 2;
    case "F8_E5M2":
    case "F8_E4M3":
    case "I8":
    case "U8":
      return 1;
    case "INT4":
    case "NF4":
    case "FP4":
    case "FP4_E2M1":
      return 0.5;
    default:
      throw new RuntimeError(`DTYPE=${dtype} NOT HANDLED`);
  }
}

export class RuntimeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RuntimeError";
  }
}
