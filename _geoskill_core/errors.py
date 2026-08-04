"""geoskill_core.errors — 错误类型与退出码契约

退出码（规划文档 §2.2）：
- 0   OK
- 2   EUSAGE: 参数/用户输入错误
- 3   EDEPEND: 依赖缺失（Python 包、外部工具）
- 4   ENETWORK: 上游网络/限流/超时
- 5   ENOMATCH: 无匹配数据 / 服务暂不可用（PHASE 0 DISABLED 也用此码）
- 6   EVALIDATE: 数据校验失败（manifest schema、CRC、unit check）
- 7   EPROCESS: 处理失败（栅格重投影、合成、指数计算等）
- 130 用户中断 (KeyboardInterrupt / SIGINT)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DEPEND = 3
EXIT_NETWORK = 4
EXIT_NOMATCH = 5
EXIT_VALIDATE = 6
EXIT_PROCESS = 7
EXIT_INTERRUPT = 130


@dataclass
class GeoSkillError(Exception):
    """所有 geoskill-core 错误的基类。

    子类必须设置 `code`（退出码）和 `kind`（错误类型短名）。
    `message` 是人类可读的错误描述。
    `details` 是结构化诊断信息（写入 stderr 时可选 JSON 化）。
    """
    message: str
    code: int = EXIT_PROCESS
    kind: str = "EGeoSkill"
    details: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[BaseException] = None

    def __str__(self) -> str:
        if self.details:
            return f"[{self.kind}] {self.message} ({self.details})"
        return f"[{self.kind}] {self.message}"


# ---- 具体错误类型 ----


class UsageError(GeoSkillError):
    """参数/用户输入错误 → exit 2"""
    def __init__(self, message: str, **details):
        super().__init__(message, code=EXIT_USAGE, kind="EUsage", details=details)


class DependencyError(GeoSkillError):
    """依赖缺失（Python 包 / 外部命令）→ exit 3"""
    def __init__(self, message: str, **details):
        super().__init__(message, code=EXIT_DEPEND, kind="EDepend", details=details)


class NetworkError(GeoSkillError):
    """网络/限流/超时 → exit 4"""
    def __init__(self, message: str, **details):
        super().__init__(message, code=EXIT_NETWORK, kind="ENetwork", details=details)


class NoMatchError(GeoSkillError):
    """无匹配数据 / 服务暂不可用 → exit 5
    PHASE 0 DISABLED 的 6 个 from-* 子命令使用此码
    """
    def __init__(self, message: str, **details):
        super().__init__(message, code=EXIT_NOMATCH, kind="ENoMatch", details=details)


class ValidationError(GeoSkillError):
    """数据校验失败 → exit 6"""
    def __init__(self, message: str, **details):
        super().__init__(message, code=EXIT_VALIDATE, kind="EValidate", details=details)


class ProcessError(GeoSkillError):
    """处理失败（栅格、合成、指数等）→ exit 7"""
    def __init__(self, message: str, **details):
        super().__init__(message, code=EXIT_PROCESS, kind="EProcess", details=details)


# ---- 退出码 → 类型 映射（用于 main() 兜底）----

EXIT_CODE_TO_TYPE: Dict[int, type] = {
    EXIT_USAGE: UsageError,
    EXIT_DEPEND: DependencyError,
    EXIT_NETWORK: NetworkError,
    EXIT_NOMATCH: NoMatchError,
    EXIT_VALIDATE: ValidationError,
    EXIT_PROCESS: ProcessError,
}


def to_exit_code(exc: BaseException) -> int:
    """把异常转为退出码。GeoSkillError 用其 code，其他走启发式。
    """
    if isinstance(exc, GeoSkillError):
        return exc.code
    if isinstance(exc, KeyboardInterrupt):
        return EXIT_INTERRUPT
    if isinstance(exc, (ValueError, TypeError, KeyError, FileNotFoundError)):
        return EXIT_USAGE
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return EXIT_DEPEND
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return EXIT_NETWORK
    return EXIT_PROCESS


__all__ = [
    "EXIT_OK", "EXIT_USAGE", "EXIT_DEPEND", "EXIT_NETWORK",
    "EXIT_NOMATCH", "EXIT_VALIDATE", "EXIT_PROCESS", "EXIT_INTERRUPT",
    "GeoSkillError", "UsageError", "DependencyError", "NetworkError",
    "NoMatchError", "ValidationError", "ProcessError",
    "EXIT_CODE_TO_TYPE", "to_exit_code",
]
