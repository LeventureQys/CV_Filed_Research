/**
 * 当前编辑器 Wendland C2 力场重建的独立参考实现。
 *
 * 输入坐标和 radius 必须使用相同单位；pressure 约定在 [0, 1]。
 * 本文件不依赖 Three.js，只重建顶点标量，不负责模型坐标变换、
 * 网格邻接平滑或最终材质着色。
 */

export function wendlandC2(normalizedDistance) {
  if (normalizedDistance >= 1) return 0;
  const remaining = 1 - normalizedDistance;
  return remaining ** 4 * (4 * normalizedDistance + 1);
}

function bucketKey(x, y, z) {
  return `${x},${y},${z}`;
}

export function buildSpatialBuckets(cells, radius) {
  if (!(radius > 0)) throw new RangeError("radius must be greater than zero");

  const buckets = new Map();
  for (const cell of cells) {
    const bucketX = Math.floor(cell.x / radius);
    const bucketY = Math.floor(cell.y / radius);
    const bucketZ = Math.floor(cell.z / radius);
    const key = bucketKey(bucketX, bucketY, bucketZ);
    const bucket = buckets.get(key);
    if (bucket) bucket.push(cell);
    else buckets.set(key, [cell]);
  }
  return { radius, buckets };
}

function queryNearbyCells(index, x, y, z) {
  const bucketX = Math.floor(x / index.radius);
  const bucketY = Math.floor(y / index.radius);
  const bucketZ = Math.floor(z / index.radius);
  const nearby = [];

  for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      for (let offsetZ = -1; offsetZ <= 1; offsetZ += 1) {
        const key = bucketKey(
          bucketX + offsetX,
          bucketY + offsetY,
          bucketZ + offsetZ,
        );
        const bucket = index.buckets.get(key);
        if (bucket) nearby.push(...bucket);
      }
    }
  }
  return nearby;
}

/**
 * 预计算顶点与 Cell 的影响关系。
 * vertices 为 [x0, y0, z0, x1, y1, z1, ...]。
 */
export function buildInfluenceCache(vertices, cells, radius) {
  if (vertices.length % 3 !== 0) {
    throw new RangeError("vertices length must be a multiple of three");
  }

  const spatialIndex = buildSpatialBuckets(cells, radius);
  const vertexCount = vertices.length / 3;
  const offsets = new Uint32Array(vertexCount + 1);
  const cellIds = [];
  const weights = [];
  const coveredVertices = [];

  for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex += 1) {
    offsets[vertexIndex] = cellIds.length;
    const x = vertices[vertexIndex * 3];
    const y = vertices[vertexIndex * 3 + 1];
    const z = vertices[vertexIndex * 3 + 2];
    const nearby = queryNearbyCells(spatialIndex, x, y, z);

    for (const cell of nearby) {
      const deltaX = x - cell.x;
      const deltaY = y - cell.y;
      const deltaZ = z - cell.z;
      const distance = Math.sqrt(
        deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ,
      );
      if (distance > radius) continue;

      const weight = wendlandC2(distance / radius);
      if (weight <= 1e-8) continue;
      cellIds.push(cell.id);
      weights.push(weight);
    }

    if (cellIds.length > offsets[vertexIndex]) {
      coveredVertices.push(vertexIndex);
    }
  }
  offsets[vertexCount] = cellIds.length;

  return {
    vertexCount,
    offsets,
    cellIds: Uint32Array.from(cellIds),
    weights: Float32Array.from(weights),
    coveredVertices: Uint32Array.from(coveredVertices),
  };
}

/**
 * 使用缓存重建标量场。
 * pressures 可以是 Map，也可以是以 Cell ID 为键的普通对象。
 */
export function reconstructScalars(cache, pressures) {
  const scalars = new Float32Array(cache.vertexCount);
  const confidences = new Float32Array(cache.vertexCount);
  const readPressure = pressures instanceof Map
    ? (cellId) => pressures.get(cellId) ?? 0
    : (cellId) => pressures[cellId] ?? 0;

  for (const vertexIndex of cache.coveredVertices) {
    const start = cache.offsets[vertexIndex];
    const end = cache.offsets[vertexIndex + 1];
    let weightedPressure = 0;
    let weightSum = 0;

    for (let influenceIndex = start; influenceIndex < end; influenceIndex += 1) {
      const weight = cache.weights[influenceIndex];
      weightedPressure += readPressure(cache.cellIds[influenceIndex]) * weight;
      weightSum += weight;
    }

    if (weightSum > 1e-6) {
      scalars[vertexIndex] = weightedPressure / weightSum;
      confidences[vertexIndex] = Math.min(1, weightSum);
    }
  }

  return { scalars, confidences };
}

