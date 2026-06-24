"""PHI 去标识化（T009，FR-007）—— 外发前抹掉个人可识别信息。

constitution 隐私铁律 + FR-007：任何文本在**进向量库 / 进日志 / 送外部 LLM**（如 T040 的
QA-conflict LLM-judge）之前，必须先去标识——电话、身份证、邮箱、带标签的姓名/住址等 PHI 一律抹成
占位符。`assert_no_phi` 给外发前的硬护栏。

高精度正则（宁可少抹也别误抹医学数值/剂量）：手机号、身份证、邮箱、（可选）日期、带标签字段
（姓名:/电话:/住址: 等）。**不动 `5mg` 这类剂量**（与 [normalize] 同红线）。纯标准库本地全测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 值型 PHI（构成"残留 PHI"判定的依据）。
_ID_CARD = re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")     # 身份证 18/15 位
_PHONE_CN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")                # 中国手机号
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_DATE = re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?")
# 带标签字段：保留标签、抹掉值。
_LABELED = re.compile(
    r"(姓名|患者姓名|患者|病人|住址|地址|联系电话|联系方式|电话|name|patient|address|phone)"
    r"(\s*[:：]\s*)(\S+)",
    re.IGNORECASE,
)


@dataclass
class DeidConfig:
    id_card: bool = True
    phone: bool = True
    email: bool = True
    dates: bool = False              # 日期常有临床意义，默认不抹
    labeled_fields: bool = True


@dataclass
class PHIMatch:
    category: str
    text: str


@dataclass
class DeidResult:
    text: str
    found: list[PHIMatch] = field(default_factory=list)


def deidentify(text: str, config: DeidConfig | None = None) -> DeidResult:
    """抹掉 PHI，返回清洗文本 + 命中清单。占位符 [ID]/[PHONE]/[EMAIL]/[DATE]/[REDACTED]。"""
    cfg = config or DeidConfig()
    found: list[PHIMatch] = []

    def _value_sub(category: str, token: str):
        def _sub(m: re.Match) -> str:
            found.append(PHIMatch(category, m.group(0)))
            return token
        return _sub

    if cfg.id_card:
        text = _ID_CARD.sub(_value_sub("id_card", "[ID]"), text)
    if cfg.phone:
        text = _PHONE_CN.sub(_value_sub("phone", "[PHONE]"), text)
    if cfg.email:
        text = _EMAIL.sub(_value_sub("email", "[EMAIL]"), text)
    if cfg.dates:
        text = _DATE.sub(_value_sub("date", "[DATE]"), text)
    if cfg.labeled_fields:
        def _lab(m: re.Match) -> str:
            if m.group(3) == "[REDACTED]":           # 幂等：已抹过的不重复记
                return m.group(0)
            found.append(PHIMatch("labeled", m.group(0)))
            return f"{m.group(1)}{m.group(2)}[REDACTED]"
        text = _LABELED.sub(_lab, text)

    return DeidResult(text=text, found=found)


def has_phi(text: str, config: DeidConfig | None = None) -> bool:
    """是否含**值型** PHI（手机号/身份证/邮箱/可选日期）。占位符不算。"""
    cfg = config or DeidConfig()
    if cfg.id_card and _ID_CARD.search(text):
        return True
    if cfg.phone and _PHONE_CN.search(text):
        return True
    if cfg.email and _EMAIL.search(text):
        return True
    if cfg.dates and _DATE.search(text):
        return True
    return False


def assert_no_phi(text: str, config: DeidConfig | None = None) -> None:
    """外发前硬护栏：仍含值型 PHI 则抛错（FR-007）。"""
    if has_phi(text, config):
        raise ValueError("文本仍含 PHI，禁止外发/入库（FR-007）")


__all__ = ["DeidConfig", "PHIMatch", "DeidResult", "deidentify", "has_phi", "assert_no_phi"]
