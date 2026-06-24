"""文本归一化（T038）—— 入库前清洗，但**医学数值/单位/剂量绝不归一**。

RAG 库的清洗有一条特殊红线（plan.md）：通用文本可以折叠空白/标点、NFKC、剔样板，但
**`5mg` ≠ `50mg`、`5mg` ≠ `5mL`**——数值/单位/剂量一旦被"归一"就是事实篡改，医学场景致命。
所以本模块只做**保信息**的清洗：NFKC（全角→半角等价折叠，不动数字含义）、空白折叠、重复标点折叠、
样板剔除（免责声明/页眉页脚）；**绝不**改写数字、单位、量纲，也不在数字与单位间动手脚。

纯标准库（unicodedata + re），本地全测。各步均为开关（可复现/可消融）。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# 默认样板模式（整行匹配即剔除）：免责声明 / 版权 / 页码页眉页脚。可按语料扩展。
_DEFAULT_BOILERPLATE = (
    r"^\s*(disclaimer|免责声明)[:：].*$",
    r"^\s*(copyright|版权所有|©).*$",
    r"^\s*(page\s*\d+|第\s*\d+\s*页)\s*$",
    r"^\s*[-=*_]{3,}\s*$",            # 分隔线
)

# 重复标点折叠（仅折叠**相同**标点的连排，不跨标点合并）。
_REPEAT_PUNCT = re.compile(r"([。！？!?.,，；;])\1+")
_WS = re.compile(r"[ \t　]+")     # 行内空白（含全角空格）
_MULTI_NL = re.compile(r"\n\s*\n\s*\n+")  # 3+ 空行 → 1 空行


@dataclass
class NormalizeConfig:
    nfkc: bool = True
    strip_boilerplate: bool = True
    collapse_whitespace: bool = True
    collapse_punct: bool = True
    boilerplate_patterns: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_BOILERPLATE)


def _strip_boilerplate(text: str, patterns: tuple[str, ...]) -> str:
    regexes = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
    lines = []
    for line in text.splitlines():
        if any(rx.match(line) for rx in regexes):
            continue
        lines.append(line)
    return "\n".join(lines)


def normalize_text(text: str, config: NormalizeConfig | None = None) -> str:
    """按开关清洗文本；**不触碰数字/单位/剂量的含义**。幂等。"""
    cfg = config or NormalizeConfig()
    if cfg.nfkc:
        # NFKC 折叠全角/兼容字符（如全角数字→半角、全角括号），不改变数值大小。
        text = unicodedata.normalize("NFKC", text)
    if cfg.strip_boilerplate:
        text = _strip_boilerplate(text, cfg.boilerplate_patterns)
    if cfg.collapse_punct:
        text = _REPEAT_PUNCT.sub(r"\1", text)
    if cfg.collapse_whitespace:
        text = _WS.sub(" ", text)
        text = _MULTI_NL.sub("\n\n", text)
        text = "\n".join(line.strip() for line in text.splitlines())
        text = text.strip()
    return text


__all__ = ["NormalizeConfig", "normalize_text"]
