"""geoskill-core — Phase 1 公共契约与 vendoring 工具

设计原则（2026-07-26）：
- 每个 skill 独立运行（不允许运行时跨 skill 依赖）
- 本包是**源仓库**：定义接口契约 + 参考实现
- 通过 `vendor.py` 把核心代码 copy 到各 skill 目录
- 通过 `tests/contract_test_*.py` 保证 vendored 副本行为一致
- 退出码契约（规划文档 §2.2）：0=成功, 2=参数错, 3=依赖缺失,
  4=网络/限流, 5=无匹配/服务暂不可用, 6=数据校验失败, 7=处理失败, 130=用户中断

模块：
- aoi: 地名 → bbox + 候选 + 置信度
- safe_download: .part、原子替换、重试、resume
- manifest: 四个 manifest 数据结构 + JSON Schema
- errors: 错误类型 + 退出码
- credentials (Phase 7, 2026-07-27): 统一凭证管理（NASA Earthdata / FIRMS / CMA / LLM / EOG）
"""

__version__ = "0.2.0"
__phase__ = "Phase 7 (2026-07-27) — added credentials"
