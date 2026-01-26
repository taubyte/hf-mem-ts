import { SafetensorsDtypes, getSafetensorsDtypeBytes } from "./types.js";

export interface DtypeMetadata {
  paramCount: number;
  bytesCount: number;
}

export interface ComponentMetadata {
  dtypes: Record<SafetensorsDtypes, DtypeMetadata>;
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
      dtypes: {} as Record<SafetensorsDtypes, DtypeMetadata>,
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
        component.dtypes[dtype as SafetensorsDtypes] = {
          paramCount: 0,
          bytesCount: 0,
        };
      }

      const dtypeBytes = getSafetensorsDtypeBytes(dtype);
      const currentShape = typedValue.shape.reduce((acc, val) => acc * val, 1);
      const currentShapeBytes = currentShape * dtypeBytes;

      component.dtypes[dtype as SafetensorsDtypes].paramCount += currentShape;
      component.dtypes[dtype as SafetensorsDtypes].bytesCount += currentShapeBytes;
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
