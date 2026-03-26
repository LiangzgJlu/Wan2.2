# Wan2.2 模型结构解析（含核心代码与作用）

> 面向仓库 `Wan2.2` 推理代码的结构化说明，重点覆盖：整体执行链路、DiT 主干、MoE 双模型切换、VAE、文本编码器与多任务封装。

---

## 1. 先看整体：Wan2.2 的“系统分层”

从代码实现看，Wan2.2 不是单一网络文件，而是一套「任务封装 + 条件编码 + 扩散主干 + 解码器 + 采样器」的组合系统。

```text
generate.py (命令行入口与任务路由)
  └── wan/<task>.py (WanT2V / WanI2V / WanTI2V / WanS2V / WanAnimate)
      ├── T5EncoderModel (文本编码)
      ├── WanModel (DiT 视频扩散主干)
      │   ├── WanAttentionBlock × N
      │   ├── Self/Cross Attention + RoPE
      │   └── Head + unpatchify
      ├── VAE (Wan2_1_VAE 或 Wan2_2_VAE)
      └── Flow Matching 采样器 (UniPC / DPM++)
```

---

## 2. 入口与任务分发

## 2.1 CLI 主入口
- **核心代码**：`generate.py`
- **作用**：解析参数、校验任务、选择配置、初始化分布式与模型实例，最终执行生成并保存视频。

关键点：
1. `--task` 决定使用哪个任务封装（如 `t2v-A14B`、`i2v-A14B`、`ti2v-5B`）。
2. `_validate_args` 会根据任务自动填充默认 prompt、步数、shift、guide scale、帧数等。
3. 支持 prompt extension（DashScope / 本地 Qwen）作为可选前处理。

## 2.2 任务与配置注册
- **核心代码**：`wan/configs/__init__.py`
- **作用**：
  - 用 `WAN_CONFIGS` 将任务名映射到具体配置。
  - 用 `SUPPORTED_SIZES` 约束每个任务支持的分辨率。
  - 用 `SIZE_CONFIGS`、`MAX_AREA_CONFIGS` 做尺寸与面积换算。

---

## 3. 配置层：模型“规格书”

以 `wan/configs/wan_t2v_A14B.py` 为例：
- **核心作用**：定义 A14B 模型结构超参数与推理超参数。
- **关键字段**：
  - `dim=5120, num_layers=40, num_heads=40`：DiT 主干规模。
  - `patch_size=(1,2,2)`：时空 patch 化粒度。
  - `low_noise_checkpoint` / `high_noise_checkpoint`：MoE 双专家权重目录。
  - `boundary=0.875`：扩散时间步的专家切换阈值。

共享配置在 `wan/configs/shared_config.py`：
- `text_len=512`、`num_train_timesteps=1000`、`param_dtype=torch.bfloat16` 等基础参数统一定义。

---

## 4. 任务封装层：推理流程编排

Wan2.2 各任务类的职责非常明确：

## 4.1 `WanT2V`（文本生成视频）
- **核心代码**：`wan/text2video.py`
- **作用**：
  1. 初始化 `T5EncoderModel`、`Wan2_1_VAE`、`WanModel`（低噪+高噪双模型）。
  2. 在 `generate()` 中执行：文本编码 → latent 初始化 → 采样迭代 → VAE 解码。
  3. 在每个时间步用 `_prepare_model_for_timestep()` 根据 `boundary` 选择 `high_noise_model` 或 `low_noise_model`，并可做 CPU/GPU offload，降低显存峰值。

## 4.2 `WanI2V`（图生视频）
- **核心代码**：`wan/image2video.py`
- **作用**：
  - 在 T2V 流程上增加图像条件分支（输入图像经过处理后注入扩散过程）。
  - 同样使用 MoE 双模型（低噪/高噪）策略。

## 4.3 `WanTI2V`（文图统一 5B）
- **核心代码**：`wan/textimage2video.py`
- **作用**：
  - 统一封装 text-to-video 与 image-to-video 两种调用。
  - 与 A14B 不同点是 **使用 `Wan2_2_VAE`**（更高压缩率实现高效 720P 推理）。

> 直观上可以理解为：`wan/<task>.py` 是“导演层”，真正的 Transformer 主干在 `wan/modules/model.py`。

---

## 5. WanModel（DiT 主干）结构拆解

- **核心代码**：`wan/modules/model.py`
- **类**：`WanModel(ModelMixin, ConfigMixin)`

## 5.1 输入表示（Patch + 条件嵌入）
1. `patch_embedding = nn.Conv3d(...)`
   - 把视频 latent `[C,F,H,W]` 切成 3D patch token。
2. `text_embedding`
   - 把文本编码器输出投影到 DiT 隐空间。
3. `time_embedding` + `time_projection`
   - 把扩散 timestep 变成调制向量（用于各层 Ada-like 调制）。

## 5.2 主干 Block：`WanAttentionBlock`
每层包含：
1. `self_attn`：视频 token 自注意力（带 RoPE）。
2. `cross_attn`：与文本 token 跨注意力融合。
3. `ffn`：前馈网络。
4. 调制参数 `modulation`：结合时间嵌入对各子层进行缩放/偏移控制。

## 5.3 位置编码（时空 RoPE）
- 关键函数：`rope_params`, `rope_apply`
- 逻辑：把时序 F 与空间 H/W 分轴编码后拼接到 Q/K，显式建模时空结构。

## 5.4 输出层与反 patch
1. `Head`：把 token 映射回 patch 像素块。
2. `unpatchify()`：将 patch 序列还原成 latent 视频张量。

---

## 6. 注意力与归一化细节

仍在 `wan/modules/model.py` 内：
- `WanRMSNorm` / `WanLayerNorm`：统一数值稳定策略。
- `WanSelfAttention`：QKV 线性层 + 可选 `qk_norm` + `flash_attention`。
- `WanCrossAttention`：文本条件融合。

另外，`wan/modules/attention.py` 提供底层高效 attention 内核封装（含 Flash Attention 路径）。

---

## 7. VAE：像素域与 latent 域桥梁

## 7.1 `Wan2_1_VAE`
- **主要使用任务**：T2V / I2V / S2V（A14B 系列）。

## 7.2 `Wan2_2_VAE`
- **主要使用任务**：TI2V-5B。
- **核心代码**：`wan/modules/vae2_2.py`
- **结构特征**：
  - `CausalConv3d`：保证时间因果一致性。
  - `Resample` + `ResidualBlock`：时空上下采样与残差建模。
  - 针对视频时序 chunk/cache 进行了推理友好优化。

---

## 8. MoE 在推理代码中的落地方式

Wan2.2 README 提到 MoE 思想；在仓库推理代码里，MoE 体现为**按扩散时间步切换双专家模型**：

- `low_noise_model`：负责后期低噪阶段细节收敛。
- `high_noise_model`：负责前期高噪阶段全局结构成形。
- `_prepare_model_for_timestep()`：依据 `boundary` 做路由；可配合 offload 在双模型之间搬运设备。

这是一种“时间维度专家分工”，在维持推理效率的同时提升容量利用率。

---

## 9. 采样器与扩散调度

- **核心代码**：
  - `wan/utils/fm_solvers.py`
  - `wan/utils/fm_solvers_unipc.py`
- **作用**：
  - 提供 Flow Matching 的采样时间步与 sigma 调度。
  - 支持 `unipc` 与 `dpm++` 两种 solver。

任务类 `generate()` 在每一步会：
1. 选择当前专家模型（A14B 为双模型）；
2. 执行 CFG（正/负条件）；
3. 用 solver 更新 latent；
4. 迭代至最终 timestep 后 VAE 解码输出视频。

---

## 10. 你最该重点读的核心代码清单（建议顺序）

1. `generate.py`：看清入口参数、任务路由与运行流程。
2. `wan/text2video.py`：理解标准 T2V + MoE 双模型切换。
3. `wan/image2video.py`：看 I2V 条件注入方式。
4. `wan/textimage2video.py`：看 TI2V-5B 如何统一 T2V/I2V。
5. `wan/modules/model.py`：核心 DiT（最重要）。
6. `wan/modules/vae2_2.py`、`wan/modules/vae2_1.py`：latent↔像素桥接。
7. `wan/utils/fm_solvers*.py`：采样器细节。
8. `wan/configs/*.py`：规格超参与默认推理参数。

---

## 11. 一句话总结

**Wan2.2 的代码结构可以概括为：以 `WanModel`（视频 DiT）为核心，外层由任务类组织多模态条件与采样循环，底层由 VAE 完成压缩域到像素域转换，并通过按时间步切换的双专家模型实现 MoE 推理。**

