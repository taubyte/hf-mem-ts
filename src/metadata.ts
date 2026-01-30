import { getSafetensorsDtypeBytes } from "./types.js";

export interface DtypeMetadata {
  paramCount: number;
  bytesCount: number;
}

export interface ComponentMetadata {
  /** Per-dtype param and byte counts (includes quantized: INT4, NF4, FP4, etc.). */
  dtypes: Record<string, DtypeMetadata>;
  paramCount: number;
  bytesCount: number;
}

export interface SafetensorsMetadata {
  components: Record<string, ComponentMetadata>;
  paramCount: number;
  bytesCount: number;
}

interface RawMetadataValue {
  dtype: string;
  shape: number[];
}

interface RawMetadata {
  [key: string]: RawMetadataValue | any;
}

export function parseSafetensorsMetadata(
  rawMetadata: Record<string, RawMetadata>
): SafetensorsMetadata {
  const components: Record<string, ComponentMetadata> = {};
  let totalParamCount = 0;
  let totalBytesCount = 0;

  for (const [name, metadata] of Object.entries(rawMetadata)) {
    const component: ComponentMetadata = {
      dtypes: {},
      paramCount: 0,
      bytesCount: 0,
    };

    for (const [key, value] of Object.entries(metadata)) {
      if (key === "__metadata__") {
        continue;
      }

      const typedValue = value as RawMetadataValue;
      const dtype = typedValue.dtype;
      
      if (!(dtype in component.dtypes)) {
        component.dtypes[dtype] = {
          paramCount: 0,
          bytesCount: 0,
        };
      }

      const dtypeBytes = getSafetensorsDtypeBytes(dtype);
      const currentShape = typedValue.shape.reduce((acc, val) => acc * val, 1);
      const currentShapeBytes = currentShape * dtypeBytes;

      component.dtypes[dtype].paramCount += currentShape;
      component.dtypes[dtype].bytesCount += currentShapeBytes;
      component.paramCount += currentShape;
      component.bytesCount += currentShapeBytes;
      totalParamCount += currentShape;
      totalBytesCount += currentShapeBytes;
    }

    components[name] = component;
  }

  return {
    components,
    paramCount: totalParamCount,
    bytesCount: totalBytesCount,
  };
}
