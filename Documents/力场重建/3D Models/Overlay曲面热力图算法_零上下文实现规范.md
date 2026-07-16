# Overlay 曲面热力图算法：零上下文实现规范

## 1. 文档目标

本文是一份可直接迁移到任意 3D 软件项目中的工程实现规范。实施者不需要了解本文产生时所依赖的项目、模型、代码或历史讨论。

目标是在任意三角曲面模型上，根据少量表面采样点实时显示连续热力图，同时满足：

- 热力图分辨率不绑定原始渲染模型的三角形密度。
- 热力沿模型表面传播，不穿过薄壁，不跨越断开的零件。
- 不直接修改原始模型的顶点、索引、材质或 UV。
- 原始模型为 `50k~1.5M` 三角形时仍可使用。
- 在常规桌面 CPU 上，缓存建立后可进行 `60 Hz` 标量更新。
- 支持局部热区、动态采样值、固定或可变采样位置。
- 标量场计算、颜色映射和原模型渲染彼此解耦。

本文默认使用：

> **独立局部 Overlay Mesh + 曲面最短路径 + Wendland C2 归一化插值 + 预计算稀疏影响缓存 + Shader 色带映射。**

---

## 2. 适用范围与非目标

### 2.1 适用对象

算法适用于定义在二维模型表面上的局部连续或分段连续标量，例如：

- 表面压力；
- 温度；
- 磨损量；
- 厚度；
- 位移幅值；
- 振动幅值；
- 置信度或质量评分。

基本假设是：沿允许传播的曲面路径距离较近的位置，其标量值通常具有相关性。

### 2.2 不属于本算法的内容

本算法不执行：

- 有限元求解；
- 材料内部应力或应变计算；
- 接触力学模拟；
- 真实三维体积场重建；
- 未被采样到的尖峰恢复；
- 物理守恒方程求解。

输出是“给定采样和传播规则后的表面插值结果”，不能自动解释为真实物理场。

---

## 3. 核心设计原则

系统必须将以下三层分开：

```text
原始渲染模型 Render Mesh
    负责几何外观、材质、遮挡和拾取

独立热力覆盖层 Overlay Mesh
    负责热力场计算采样位置和显示分辨率

热力标量与色带 Scalar + LUT
    负责实时数值和最终颜色
```

不得把原始渲染模型同时作为高模显示、曲面传播图、热力分辨率和动态颜色缓存的唯一载体。

### 3.1 保存标量，不保存最终 RGB

每个 Overlay 顶点保存一个 `float` 热力标量。颜色只在 Fragment Shader 中通过一维 LUT 生成。

禁止每帧在 CPU 上生成并上传 RGB/RGBA 顶点颜色，除非目标引擎完全不支持标量属性和 Shader LUT。

### 3.2 几何预处理与实时更新分离

只有模型、Overlay、采样位置、传播半径或边界规则变化时，才重建几何和影响缓存。

采样值变化时，每帧只进行稀疏加权和。

### 3.3 所有曲面计算使用模型局部坐标

Overlay、采样点、距离和半径统一在模型局部坐标中计算。模型的世界变换只用于渲染。

如果输入采样点是世界坐标，必须先乘以模型世界矩阵的逆矩阵转换到局部坐标。

必须明确模型单位。推荐内部统一为毫米或米，`supportRadius` 和焊接容差必须使用相同单位。

---

## 4. 术语

| 术语 | 含义 |
|---|---|
| Render Mesh | 原始渲染模型，可为 50k~1.5M 或更多三角形 |
| Overlay Mesh | 独立于原模型的局部细密热力覆盖网格 |
| Sample / Cell | 一条表面采样，包含位置、法线和标量值 |
| ROI | Region of Interest，允许显示热力图的目标表面区域 |
| Support Radius | 单个采样沿曲面传播的最大路径半径 |
| Coverage | 某位置受到采样覆盖的强度或置信度 |
| Influence Cache | Overlay 顶点与采样之间预计算的稀疏权重缓存 |
| Barrier Edge | 不允许热力传播跨越的网格边 |
| Welded Vertex | 合并几何重合顶点后得到的拓扑顶点 |
| LUT | 一维颜色查找表，例如 256 项 RGBA 纹理 |

---

## 5. 输入与输出

### 5.1 原始模型输入

最低要求：

```cpp
struct RenderMeshInput {
    Array<Vec3> positionsLocal;
    Array<uint32_t> triangleIndices; // 每 3 个索引组成一个三角形
    Optional<Array<Vec3>> normalsLocal;
    Mat4 modelToWorld;
    float unitScaleToMeters;
};
```

要求：

- `positionsLocal` 和索引在构建期间不可变化。
- 允许存在重复顶点、UV 接缝和硬法线顶点。
- 允许存在多个连通分量。
- 非流形、自交和退化三角形必须检测并记录。

### 5.2 采样输入

```cpp
struct SurfaceSample {
    uint64_t id;
    Vec3 positionLocal;
    Vec3 normalLocal;
    float value;
    bool enabled;
};
```

推荐额外保存稳定附着信息：

```cpp
struct SurfaceAttachment {
    uint32_t sourceTriangle;
    Vec3 barycentric;       // 三个分量之和为 1
    Vec3 surfacePointLocal;
    Vec3 surfaceNormalLocal;
    float attachmentError;
};
```

如果采样数据已经包含 `sourceTriangle + barycentric`，优先使用，避免每次重新搜索最近面。

### 5.3 配置输入

```cpp
struct HeatOverlayConfig {
    float supportRadius;
    float roiMargin;
    float weldToleranceRatio = 1e-7f;
    float foldPenalty = 2.0f;
    float maxCrossingAngleDegrees = 180.0f;
    float targetOverlayEdgeLength;
    uint32_t maxInfluencesPerVertex = 8;
    float minValue = 0.0f;
    float maxValue = 1.0f;
    float overlayDepthBias;
    bool respectConnectedComponents = true;
    bool useHardAngleBarrier = false;
};
```

### 5.4 输出

构建阶段输出：

```cpp
struct HeatOverlayAsset {
    OverlayMesh overlay;
    InfluenceCache influenceCache;
    Array<SurfaceAttachment> sampleAttachments;
    Bounds localBounds;
    uint64_t geometryVersion;
    uint64_t cacheVersion;
};
```

实时阶段输出：

```cpp
Array<float> effectiveScalarPerOverlayVertex;
```

上传 GPU 后由热力 Overlay Draw Call 显示。

---

## 6. 总体生命周期

```text
加载原始模型
    ↓
构建焊接拓扑和加速结构
    ↓
采样附着到曲面
    ↓
确定目标表面 ROI
    ↓
生成独立 Overlay Mesh
    ↓
建立 Overlay 邻接图和传播边界
    ↓
预计算 Sample → Overlay 顶点曲面距离
    ↓
计算 Wendland 权重并压缩为 Influence Cache
    ↓
创建 GPU Overlay VBO/IBO 与标量 Buffer
    ↓
每帧读取 Sample value
    ↓
稀疏加权更新 Overlay 标量
    ↓
上传单通道标量
    ↓
Fragment Shader 查询 LUT
```

---

## 7. 阶段 A：原始模型拓扑预处理

### 7.1 清理退化三角形

三角形面积：

```text
area2 = length(cross(p1 - p0, p2 - p0))
```

若 `area2 <= epsilonArea`，跳过该三角形并记录统计。`epsilonArea` 应根据包围盒尺度设置，不要写死绝对值。

### 7.2 顶点焊接

原始 STL、CAD 三角化和硬法线模型经常在相同几何位置保存多个顶点。曲面传播图不能直接使用这些展开顶点。

推荐焊接容差：

```text
boundsDiagonal = length(boundsMax - boundsMin)
weldTolerance = max(
    boundsDiagonal * clamp(weldToleranceRatio, 1e-9, 1e-4),
    machineSafeMinimum
)
```

默认 `weldToleranceRatio = 1e-7`。

实现方式：

1. 使用 3D 空间哈希或量化网格。
2. 将位置除以 `weldTolerance` 后取整，作为桶坐标。
3. 检查当前桶和相邻 26 个桶。
4. 仅合并实际距离不大于容差的顶点。
5. 保存 `renderVertexToWeldedVertex` 映射。

禁止为了修复裂缝而使用过大容差，否则会错误合并薄壁两侧。

### 7.3 共享边和邻接

对每个焊接三角形插入无向边：

```text
edgeKey = (min(v0,v1), max(v0,v1))
```

记录：

- 边的两个端点；
- 相邻三角形；
- 边长；
- 两侧面法线；
- 是否为边界边；
- 是否非流形；
- 是否是用户或业务定义的 Barrier。

### 7.4 连通分量

按共享边计算三角形或顶点连通分量。

- 不同连通分量之间严格禁止传播。
- 如果模型由多个零件组成，这一步防止空间相邻零件互相串色。

### 7.5 加速结构

必须为原始三角形建立 BVH、AABB Tree 或等价空间索引，用于：

- 采样点最近面查询；
- Overlay 顶点投影；
- 用户拾取；
- 高模与简化网格之间的映射。

不得使用遍历全部三角形的最近点查询作为生产实现。

---

## 8. 阶段 B：采样附着到曲面

### 8.1 最近点候选

对每个采样位置使用 BVH 查询若干最近三角形候选，并计算点到三角形的最近点及重心坐标。

### 8.2 法线消歧

空间上相邻的薄壁两侧可能距离相似。候选评分推荐为：

```text
distanceScore = distance / distanceTolerance
normalScore = 1 - clamp(dot(normalize(sampleNormal), triangleNormal), -1, 1)
score = distanceScore + normalWeight * normalScore
```

如果采样法线方向可能正反不确定，可使用：

```text
abs(dot(sampleNormal, triangleNormal))
```

但只有业务明确允许时才能使用绝对值。压力传感器通常应保留法线方向语义。

### 8.3 附着有效性

若最近距离超过业务允许的 `maxAttachmentDistance`，该采样无效，不得强行吸附。

无效采样：

- 不进入影响缓存；
- 记录一次 warning；
- UI 可显示未附着标识；
- 不得导致整个 Overlay 构建失败。

---

## 9. 阶段 C：确定目标表面 ROI

Overlay 不应默认覆盖完整高模。推荐从采样附着三角形出发，沿曲面拓扑扩展目标区域。

### 9.1 默认 ROI

从全部有效采样附着位置进行多源 Dijkstra，扩展半径：

```text
roiRadius = supportRadius + roiMargin
```

推荐：

```text
roiMargin = max(0.25 * supportRadius, 2 * targetOverlayEdgeLength)
```

收集路径距离不超过 `roiRadius` 的三角形，形成 ROI。

### 9.2 传播边界

以下边默认不得跨越，或必须由配置明确允许：

- 不同连通分量之间；
- 非流形边；
- 用户标记的物理边界；
- 材料分区边界；
- 裂纹、密封、隔热或接触分区边界；
- 超过硬折角阈值的边。

### 9.3 折角代价

如果边允许跨越，但希望抑制热量翻过尖角，使用软折角代价：

```text
theta = acos(clamp(abs(dot(normal0, normal1)), 0, 1))
edgeCost = edgeLength * (1 + foldPenalty * (theta / pi)^2)
```

默认 `foldPenalty = 2.0`。

注意：折角惩罚不能替代明确的物理边界。真实不连续边必须设置为 Barrier。

---

## 10. 阶段 D：生成独立 Overlay Mesh

Overlay 是一张独立局部网格。它贴合原模型 ROI，但拥有自己的顶点密度、索引、标量属性和 Draw Call。

### 10.1 MVP 推荐算法：提取、细分、投影

这是最容易稳定实现的通用版本。

1. 从原始焊接网格提取 ROI 三角形。
2. 复制 ROI 为独立 Overlay Mesh。
3. 根据目标边长递归细分过大的三角形。
4. 新顶点首先按父边或父三角形插值生成。
5. 使用原始模型 BVH 将新顶点投影回允许的源表面。
6. 保存每个 Overlay 顶点对应的源三角形和重心坐标。
7. 重新计算 Overlay 顶点法线和邻接图。
8. 删除极小、翻转或跨越 Barrier 的三角形。

### 10.2 目标边长

Overlay 分辨率应由场的空间变化和屏幕需求决定，不由原始高模面数决定。

建议初始值：

```text
targetOverlayEdgeLength <= min(
    supportRadius / 6,
    medianNearestSampleSurfaceDistance / 3
)
```

如果只用于视觉显示，可再结合屏幕误差限制。

典型单个活动 Overlay：

- `10k~50k` 顶点；
- `20k~100k` 三角形；
- 每个顶点最多缓存 `4~8` 个主要采样影响。

### 10.3 自适应细分

优先细分：

- 高曲率区域；
- 采样附近；
- 预计标量梯度高的区域；
- 屏幕投影面积大的区域；
- 原始三角形过大的区域。

低曲率、低梯度和远离采样的位置可以保持较粗。

### 10.4 Overlay 到原模型的映射

每个 Overlay 顶点必须保存：

```cpp
struct OverlaySourceBinding {
    uint32_t sourceTriangle;
    Vec3 barycentric;
};
```

原模型不变时，Overlay 位置可直接存储。

如果原模型会骨骼变形、节点形变或有限元变形，应每帧通过源三角形三个变形后顶点和重心坐标重建 Overlay 位置及法线。

### 10.5 防止 Z-fighting

优先顺序：

1. 使用 Rasterizer Depth Bias / Polygon Offset。
2. 在 Vertex Shader 中沿表面法线偏移极小距离。
3. 使用 `LessEqual` 深度测试并关闭深度写入。

法线偏移量应相对模型尺度设置，例如包围盒对角线的 `1e-5~1e-4`，并设置上下限。

不得为了消除 Z-fighting 大幅偏移 Overlay，否则轮廓会漂浮。

---

## 11. 阶段 E：Overlay 曲面传播图

Overlay 本身建立独立邻接图。

每条边保存：

```cpp
struct OverlayGraphEdge {
    uint32_t neighbor;
    float cost;
    bool barrier;
};
```

边代价默认使用 Overlay 边长和折角惩罚。

如果 Overlay 来源于可靠的 ROI 拓扑，传播图直接使用 Overlay 共享边。禁止根据三维空间最近邻连接图，否则会重新引入薄壁穿透。

---

## 12. 阶段 F：预计算采样影响

### 12.1 从三角形内部初始化最短路

采样通常位于 Overlay 三角形内部，而不在顶点上。不得只选择一个最近顶点作为唯一 Dijkstra 起点。

设附着三角形三个顶点为 `v0,v1,v2`，采样表面点为 `p`，初始化：

```text
dist[v0] = length(p - position[v0])
dist[v1] = length(p - position[v1])
dist[v2] = length(p - position[v2])
```

将三个顶点一起压入优先队列。

这样可以降低网格密度和顶点位置对传播结果的影响。

### 12.2 截断 Dijkstra

每个采样只计算到 `supportRadius`：

```pseudo
function truncatedDijkstra(graph, attachment, supportRadius):
    distance = INF
    queue = MinHeap()

    for vertex in attachment.triangle.vertices:
        initial = length(attachment.point - vertex.position)
        if initial < supportRadius:
            distance[vertex] = initial
            queue.push(initial, vertex)

    while queue not empty:
        currentDistance, vertex = queue.popMin()
        if currentDistance != distance[vertex]:
            continue
        if currentDistance >= supportRadius:
            continue

        for edge in graph[vertex]:
            if edge.barrier:
                continue
            candidate = currentDistance + edge.cost
            if candidate < distance[edge.neighbor] and candidate < supportRadius:
                distance[edge.neighbor] = candidate
                queue.push(candidate, edge.neighbor)

    return onlyFiniteDistances(distance)
```

### 12.3 Wendland C2 权重

对曲面路径距离 `d`：

```text
r = d / supportRadius
```

```text
phi(r) = (1-r)^4 * (4r+1),  0 <= r < 1
phi(r) = 0,                 r >= 1
```

性质：

- 中心权重为 1；
- 支撑半径外严格为 0；
- 边界处平滑回到 0；
- 权重非负；
- 适合局部稀疏缓存。

### 12.4 归一化插值

对于 Overlay 顶点 `v`：

```text
weightSum(v) = Σ phi_i(v)
P(v) = Σ [sampleValue_i * phi_i(v)] / weightSum(v)
C(v) = min(1, weightSum(v))
E(v) = minValue + [P(v) - minValue] * C(v)
```

其中：

- `P` 是邻近采样的归一化估计；
- `C` 是覆盖强度；
- `E` 是用于颜色显示的有效标量。

必须使用归一化平均。禁止直接使用：

```text
Σ(sampleValue * weight)
```

否则采样越密集的位置会产生没有数据依据的伪峰。

### 12.5 Top-K 影响截断

一个 Overlay 顶点可能受到很多采样影响。为保证实时性能，可只保留权重最大的 `K` 项，默认 `K=8`。

正确处理顺序：

1. 先收集全部非零影响。
2. 记录完整 `weightSumAll`，用于 coverage。
3. 选择最大的 K 个权重。
4. 对保留权重重新归一化，使其和为 1。
5. 保存 `coverage = min(1, weightSumAll)`。

这样每帧只需要最多 K 次乘加，同时尽量保留原 coverage 语义。

必须通过误差测试确认 K 足够。推荐比较 K=4、8、16 与不截断结果的最大误差和 RMSE。

---

## 13. Influence Cache 数据结构

推荐使用 CSR 形式：

```cpp
struct InfluenceCache {
    Array<uint32_t> offsets;        // vertexCount + 1
    Array<uint32_t> sampleIndices;  // influenceCount
    Array<float> normalizedWeights; // influenceCount
    Array<float> coverage;          // vertexCount
};
```

对于 Overlay 顶点 `v`：

```text
begin = offsets[v]
end   = offsets[v+1]
```

影响项位于 `[begin,end)`。

### 13.1 固定 K 的替代结构

如果目标平台更适合 SIMD/GPU，且每顶点固定最多 K 项，可使用：

```cpp
struct PackedVertexInfluence {
    uint32_t sampleIndex[K];
    Half normalizedWeight[K];
    Half coverage;
    uint8_t count;
};
```

固定 K 结构浪费少量空间，但更利于连续访问和 GPU Compute。

### 13.2 精度

- CPU 权重建议构建时使用 `float32` 或 `float64`。
- 缓存可使用 `float16`，但必须验证误差。
- 实时标量输出通常可使用 `R16F`。
- 定量分析、跨度很大或需要精确读取时使用 `R32F`。

---

## 14. 每帧 CPU 更新

### 14.1 基础伪代码

```pseudo
function updateOverlayScalars(cache, sampleValues, minValue):
    parallel_for vertex in [0, overlayVertexCount):
        weightedValue = 0

        for item in cache.offsets[vertex] .. cache.offsets[vertex+1]:
            sampleIndex = cache.sampleIndices[item]
            weightedValue += sampleValues[sampleIndex] * cache.normalizedWeights[item]

        coverage = cache.coverage[vertex]
        output[vertex] = minValue + (weightedValue - minValue) * coverage

    return output
```

若顶点没有影响项：

```text
output[vertex] = minValue
```

### 14.2 C++ 实现要求

- 输入输出使用连续数组。
- 避免每帧内存分配。
- `sampleValues` 使用紧凑数组，不在内循环查询哈希表。
- 使用线程池并行，而不是每帧创建线程。
- 顶点数低于并行阈值时使用单线程，避免调度成本。
- 可按 4k~16k 顶点划分任务块。
- 固定 K 版本可使用 SIMD。
- 使用双缓冲或持久映射 Buffer，避免 CPU/GPU 同步等待。

### 14.3 60 Hz 预算

60 Hz 总帧预算：

```text
16.67 ms
```

建议 Overlay 子系统预算：

| 项目 | 建议 P95 |
|---|---:|
| 标量 CPU 更新 | <= 2.0 ms，复杂场景 <= 4.0 ms |
| GPU 上传 | <= 1.0 ms |
| Overlay Draw GPU | <= 1.0 ms |
| 热力子系统合计 | 优先 <= 4.0 ms，最大 <= 6.0 ms |

推荐单个活动 Overlay 控制在 `10k~50k` 顶点。多个 Overlay 应按总活动顶点数预算，而不是只看单个 Overlay。

---

## 15. GPU 更新可选方案

当 Overlay 顶点很多、活动 Overlay 很多或 CPU 预算紧张时，可把每帧稀疏加权移到 Compute Shader。

上传：

- `sampleValues`；
- `offsets`；
- `sampleIndices`；
- `normalizedWeights`；
- `coverage`。

输出：

- `effectiveScalar` Storage Buffer。

Compute Shader 每个线程处理一个 Overlay 顶点。几何和权重缓存仍然只在配置变化时重建。

如果只有采样值变化，每帧真正需要从 CPU 上传的可能只有几十或几百个 `float`。

---

## 16. GPU 渲染规范

### 16.1 顶点数据

Overlay VBO 最低包含：

```text
positionLocal : float3
normalLocal   : float3
heatScalar    : half/float
```

也可把 `heatScalar` 放在独立动态 Buffer 中，避免更新静态位置和法线。

### 16.2 Vertex Shader 示例

```glsl
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in float inHeat;

uniform mat4 model;
uniform mat4 viewProjection;
uniform float normalOffset;

out float heat;
out vec3 worldNormal;

void main() {
    vec3 localPosition = inPosition + inNormal * normalOffset;
    vec4 worldPosition = model * vec4(localPosition, 1.0);
    gl_Position = viewProjection * worldPosition;
    heat = inHeat;
    worldNormal = normalize(mat3(transpose(inverse(model))) * inNormal);
}
```

### 16.3 Fragment Shader 示例

```glsl
in float heat;
in vec3 worldNormal;

uniform sampler1D heatLut;
uniform float minValue;
uniform float maxValue;
uniform float opacity;

out vec4 fragColor;

void main() {
    float safeRange = max(maxValue - minValue, 1e-12);
    float t = clamp((heat - minValue) / safeRange, 0.0, 1.0);
    vec4 color = texture(heatLut, t);
    fragColor = vec4(color.rgb, color.a * opacity);
}
```

### 16.4 LUT

推荐使用 256 项一维纹理。模型和图例必须使用同一 LUT、同一 `minValue/maxValue` 和同一归一化函数。

避免默认使用 Jet。推荐：

- Viridis；
- Cividis；
- Inferno；
- 灰底→蓝→青→黄→橙→红的工程色带。

### 16.5 透明度模式

定量模式：

- Overlay 尽量不使用 coverage 再次调节 alpha；
- coverage 已进入 `E`；
- LUT 下限颜色表示无覆盖或最小值。

叠加在原模型材质上的展示模式：

- 可额外保存 `coverage` 作为 alpha；
- 必须明确这是视觉混合模式；
- 图例应说明低 coverage 区域会与底色混合，颜色不再是纯 LUT 色。

禁止把 coverage 先乘入标量，又在 RGB 阶段与固定灰色重复混合，却仍声称颜色可直接定量读取。

### 16.6 深度与混合

推荐：

```text
Depth Test: LessEqual
Depth Write: Off
Cull Mode: 与原模型一致，必要时双面
Blend: Alpha Blend 或 Opaque Quantitative Mode
Polygon Offset / Depth Bias: Enabled
```

Overlay 应在原模型不透明 Pass 之后、透明 UI 之前绘制。

---

## 17. 缓存失效规则

必须集中管理缓存版本，禁止各模块隐式判断。

| 变化 | 重建 Overlay | 重建传播图 | 重建 Influence Cache | 仅更新标量 |
|---|---:|---:|---:|---:|
| 采样 value | 否 | 否 | 否 | 是 |
| LUT/图例范围 | 否 | 否 | 否 | 可不更新，只改 Shader uniform |
| 模型世界变换 | 否 | 否 | 否 | 否 |
| 采样位置/法线 | 视 ROI 而定 | 视情况 | 是 | 否 |
| supportRadius | 可能 | 否 | 是 | 否 |
| foldPenalty | 否 | 是 | 是 | 否 |
| Barrier 边界 | 可能 | 是 | 是 | 否 |
| Overlay 分辨率 | 是 | 是 | 是 | 否 |
| 原始模型拓扑 | 是 | 是 | 是 | 否 |
| 原始模型静态顶点位置 | 是 | 是 | 是 | 否 |

### 17.1 序列化缓存

可将 Overlay 和 Influence Cache 保存到项目文件。缓存头至少包含：

```text
formatVersion
sourceMeshContentHash
sampleLayoutHash
configHash
overlayVertexCount
overlayTriangleCount
influenceCount
endianness
```

任一 Hash 不一致时拒绝使用旧缓存。

---

## 18. 多线程与异步构建

Overlay 构建和 Dijkstra 不能阻塞渲染主线程。

推荐状态机：

```text
Idle
→ AttachingSamples
→ ExtractingROI
→ BuildingOverlay
→ BuildingInfluences
→ UploadPending
→ Ready
```

要求：

- 后台任务带 `generationId`。
- 用户再次修改配置时，取消或废弃旧任务。
- 只有最新 `generationId` 的结果可以提交 GPU。
- GPU 资源创建和销毁在渲染线程执行。
- 构建期间继续显示旧 Overlay 或隐藏热力图，不阻塞相机交互。
- 报告阶段进度和错误。

---

## 19. 异常和降级行为

| 异常 | 预期行为 |
|---|---|
| 空模型 | 返回失败，不崩溃 |
| 无有效三角形 | 返回失败并记录错误 |
| 无采样 | 隐藏 Overlay 或显示最小值 |
| 单个采样无法附着 | 跳过该采样并 warning |
| 全部采样无法附着 | Overlay 不进入 Ready |
| 非流形边 | 默认 Barrier，并记录统计 |
| Overlay 投影失败 | 删除局部坏三角形；超过阈值则构建失败 |
| `maxValue <= minValue` | Shader 使用安全范围 1，避免除零 |
| NaN/Inf 采样 | 该帧按 `minValue` 或禁用处理，并记录限频 warning |
| GPU Buffer 创建失败 | 保留 CPU 缓存，热力层隐藏，原模型继续工作 |
| 后台构建被取消 | 安全释放临时资源，不提交半成品 |

不得因为热力图失败而使原始模型、拾取、相机或主程序不可用。

---

## 20. 数值验证

### 20.1 常量保持

将所有有效采样值设为同一个 `p0`。

覆盖充分区域内应满足：

```text
P(v) ≈ p0
```

不得因附近采样数量增加产生高于 `p0` 的峰。

### 20.2 有界性

权重非负时，覆盖区域的 `P(v)` 应在影响采样的最小值和最大值之间，允许浮点误差。

### 20.3 支撑边界

采样曲面距离达到 `supportRadius` 时，Wendland 权重应平滑降为 0；半径外不得存在该采样影响项。

### 20.4 coverage

- 采样中心或密集覆盖区接近 1；
- 支撑边界附近连续下降；
- 不得出现 NaN、Inf 或负值。

### 20.5 Top-K 误差

使用同一场景比较完整影响与 K=4、8、16：

```text
maxAbsoluteError
meanAbsoluteError
RMSE
```

推荐默认验收：K=8 相对完整结果的归一化 RMSE 小于 `1%`；如果不满足，提高 K。

---

## 21. 几何验证场景

实施者必须建立以下自动或半自动测试资产。

### 21.1 平面

- 规则平面；
- 采样规则排布；
- 测地距离应接近平面欧氏距离；
- 用于验证基础权重和连续性。

### 21.2 圆柱面

- 两点弦长小于圆弧长度；
- 曲面路径必须沿圆弧传播；
- 用于验证非平面测地语义。

### 21.3 U 形薄壁

- 两侧空间距离很近；
- 沿表面路径很远；
- 一侧采样不得直接影响另一侧。

### 21.4 两个断开零件

- 两零件空间上接近；
- 连通分量不同；
- 不得跨零件传播。

### 21.5 90° 折角

- 比较 `foldPenalty=0` 和 `foldPenalty>0`；
- 跨角路径代价必须增加；
- 设置硬 Barrier 后影响必须完全停止。

### 21.6 重复顶点接缝

- 几何连续但渲染顶点因 UV/法线重复；
- 焊接拓扑后传播应连续；
- 不得在接缝处断色。

### 21.7 非流形模型

- 系统应检测并默认阻断可疑边；
- 不崩溃；
- 输出诊断统计。

---

## 22. 视觉验收

所有方法比较必须固定：

- 相同采样位置和值；
- 相同 `minValue/maxValue`；
- 相同 LUT；
- 相同相机；
- 相同光照；
- 相同目标区域；
- 相同输出分辨率。

合格结果：

- 看不到逐面常色造成的随机色块；
- Overlay 边缘不出现明显悬浮和 Z-fighting；
- 热区沿曲面连续；
- UV/硬法线接缝不导致传播断裂；
- 不应跨越断开零件或明确 Barrier；
- 图例颜色与模型颜色一致；
- 改变相机不改变标量场分布。

---

## 23. 性能验收

### 23.1 测试模型档位

至少覆盖：

- 50k 三角形；
- 500k 三角形；
- 1.5M 三角形。

热力 Overlay 单独测试：

- 10k 顶点；
- 50k 顶点；
- 100k 顶点。

采样数量：

- 4；
- 16；
- 64；
- 256。

### 23.2 必须记录的指标

```text
meshPreprocessMs
sampleAttachmentMs
roiExtractionMs
overlayBuildMs
influenceCacheBuildMs
fieldUpdateCpuMs
gpuUploadMs
overlayDrawGpuMs
totalFrameCpuMs
totalFrameGpuMs
cacheMemoryBytes
overlayGpuMemoryBytes
```

### 23.3 统计方式

- Release 构建；
- 目标硬件；
- 连续运行至少 60 秒；
- 记录 P50/P95/P99；
- 不只报告平均 FPS；
- 构建阶段和实时阶段分别报告；
- 使用 GPU Timestamp Query 测 Draw GPU 时间；
- CPU 计时使用单调高精度时钟。

### 23.4 60 Hz 最低验收

在目标代表场景中：

```text
totalFrameP95 < 16.67 ms
fieldUpdateCpuP95 <= 4.0 ms
gpuUploadP95 <= 1.0 ms
overlayDrawGpuP95 <= 1.5 ms
```

如果热力数据只以 20~30 Hz 到达，可以 20~30 Hz 更新标量，渲染仍保持 60 Hz，并在前后两帧标量之间进行时间插值。

---

## 24. 推荐模块接口

```cpp
class IHeatOverlaySystem {
public:
    virtual BuildHandle buildAsync(
        const RenderMeshInput& mesh,
        Span<const SurfaceSample> samples,
        const HeatOverlayConfig& config) = 0;

    virtual void cancelBuild(BuildHandle handle) = 0;

    virtual bool commitCompletedBuild(BuildHandle handle) = 0;

    virtual void updateSampleValues(
        Span<const uint64_t> sampleIds,
        Span<const float> values) = 0;

    virtual void updateFrame(float deltaTime) = 0;

    virtual void render(const CameraData& camera) = 0;

    virtual HeatOverlayStats getStats() const = 0;

    virtual void clear() = 0;
};
```

建议拆分内部模块：

```text
MeshTopologyBuilder
SurfaceBvh
SampleAttachmentSolver
RoiExtractor
OverlayMeshBuilder
SurfaceDistanceSolver
InfluenceCacheBuilder
HeatFieldUpdater
HeatOverlayGpuResource
HeatOverlayRenderer
HeatOverlaySerializer
```

---

## 25. 实施顺序

### 阶段 1：最小可用版本

1. 读取单个静态三角模型。
2. 顶点焊接、共享边和连通分量。
3. Sample 最近面附着。
4. 从 Sample 周围提取 ROI。
5. ROI 复制并均匀细分为 Overlay。
6. Overlay 上截断 Dijkstra。
7. Wendland C2 + 归一化 + coverage。
8. CSR Influence Cache。
9. CPU 每帧标量更新。
10. 独立 VBO 标量 + Shader LUT。
11. 平面、圆柱、薄壁和断开零件测试。

### 阶段 2：生产化

1. BVH 和空间哈希优化。
2. 后台构建、取消和版本管理。
3. Top-K 影响压缩。
4. Buffer 双缓冲或持久映射。
5. Overlay 自适应细分。
6. Barrier 编辑和持久化。
7. 缓存序列化。
8. 完整性能埋点。

### 阶段 3：高级功能

1. GPU Compute 标量更新。
2. 动态变形模型绑定。
3. 多 Overlay 调度和 LOD。
4. 各向异性核。
5. Heat Method 或更精确测地距离。
6. 局部 UV 纹理高质量静态输出。

---

## 26. 常见错误清单

实施时必须避免：

1. 每帧重新运行 Dijkstra。
2. 每帧重新生成 Overlay 或 UV。
3. 用三维欧氏距离替代任意曲面的表面路径。
4. 用空间最近邻连接 Overlay 图。
5. 直接对采样贡献求和而不归一化。
6. 平滑 RGB，而不是平滑或插值标量。
7. 把原模型 UV 接缝顶点直接当作传播拓扑。
8. 使用过大焊接容差连接薄壁两侧。
9. 只从采样最近的单个顶点启动 Dijkstra。
10. 对完整 1.5M 三角形模型每帧更新顶点颜色。
11. 用 coverage 两次改变颜色，却仍用同一图例定量解释。
12. 在主线程同步构建 Overlay 和影响缓存。
13. 只看平均 FPS，不统计 P95/P99。
14. 将 Overlay 构建失败扩散为原模型渲染失败。

---

## 27. 最终实现判定

一个实现只有同时满足以下条件，才可以认为完成：

- Overlay 是独立网格，不修改原始高模拓扑。
- Overlay 分辨率可以独立配置。
- 采样附着具有三角形和重心坐标。
- 传播基于共享边曲面拓扑。
- 不同连通分量不传播。
- 支持折角代价和显式 Barrier。
- 使用 Wendland C2 紧支撑核。
- 使用归一化加权平均，不产生采样密度伪峰。
- coverage 进入最终有效标量。
- 影响关系预计算并压缩缓存。
- 采样值变化时不重建几何和距离。
- 每帧只更新单通道标量。
- Fragment Shader 使用统一 LUT。
- Overlay 使用深度偏移避免 Z-fighting。
- 构建在后台执行且可取消。
- 平面、圆柱、薄壁、断开零件和接缝测试通过。
- 目标硬件整帧 P95 满足性能要求。

核心架构可以压缩为：

```text
高模只负责显示和遮挡
        ↓
局部 Overlay 提供独立热力分辨率
        ↓
曲面拓扑决定采样影响关系
        ↓
预计算权重缓存隔离昂贵几何计算
        ↓
每帧只更新 float 标量
        ↓
Shader 在像素级插值并查颜色 LUT
```

这套架构的关键价值不是完全摆脱三角形渲染，而是让热力场的计算密度、更新成本和显示质量不再由原始高模的三角形数量直接决定。
