"""双语医学 NER（T041）—— 中英分治抽取 → 统一实体 schema。

**为什么要统一 schema**（plan.md）：英文走 scispaCy/medspaCy(+UMLS linker)、中文走 CMeEE/CCKS
训练的医疗 BERT-NER，两边标签体系不同。若不把中英实体类型**对齐到统一 schema**
（drug/disease/anatomy/finding/procedure），下游的覆盖率统计（T042）与实体 F1（T050）就不可比。

语种分治：纯启发式 `detect_language`（CJK vs 拉丁字符占比），无需第三方 langid。两个 NER 后端守卫
导入（spaCy / transformers），**可注入** `en_backend`/`zh_backend` 桩，故语种分发 + 标签映射本地全测，
真实模型在 AutoDL。`as_entity_extractor` 把本模块接到 [eval.metrics_report]（T050）的 `ner_fn`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

# 统一实体类型（中英对齐的唯一事实源）。
DRUG = "drug"
DISEASE = "disease"
ANATOMY = "anatomy"
FINDING = "finding"
PROCEDURE = "procedure"
OTHER = "other"
UNIFIED_LABELS = (DRUG, DISEASE, ANATOMY, FINDING, PROCEDURE, OTHER)

# 英文模型标签 → 统一（scispaCy bc5cdr / medspaCy / i2b2 problem-treatment-test 等常见标签）。
_EN_MAP = {
    "chemical": DRUG, "drug": DRUG, "simple_chemical": DRUG, "medication": DRUG,
    "disease": DISEASE, "disorder": DISEASE, "problem": DISEASE,
    "anatomy": ANATOMY, "anatomical_structure": ANATOMY, "body_part": ANATOMY, "organ": ANATOMY,
    "sign_symptom": FINDING, "finding": FINDING, "symptom": FINDING,
    "procedure": PROCEDURE, "treatment": PROCEDURE, "test": PROCEDURE,
}
# 中文模型标签 → 统一（CMeEE：dru/dis/bod/sym/pro/ite；及中文别名）。
_ZH_MAP = {
    "dru": DRUG, "药物": DRUG, "药品": DRUG,
    "dis": DISEASE, "疾病": DISEASE,
    "bod": ANATOMY, "身体": ANATOMY, "部位": ANATOMY,
    "sym": FINDING, "临床表现": FINDING, "症状": FINDING,
    "pro": PROCEDURE, "医疗程序": PROCEDURE, "手术": PROCEDURE,
    "ite": PROCEDURE, "医学检验": PROCEDURE,
}

_CJK = re.compile(r"[一-鿿]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")

# 后端抽取返回的原始 span：(文本, 原始标签, 起, 止)。
RawSpan = tuple[str, str, int, int]
Backend = Callable[[str], Sequence[RawSpan]]


@dataclass(frozen=True)
class Entity:
    """统一医学实体。`cui` 为 UMLS 可链接 id（可链接率=质量信号，T042）。"""

    text: str
    label: str               # 统一 schema 之一
    start: int = -1
    end: int = -1
    lang: str | None = None
    cui: str | None = None
    score: float = 1.0

    @property
    def norm(self) -> str:
        return self.text.strip().lower()


def detect_language(text: str) -> str:
    """启发式语种判定：CJK 字符数 > 拉丁**词**数 → 'zh'，否则 'en'。

    用拉丁词数（非字母数）作比，使"大段中文 + 个别英文借词"仍判为 zh。
    """
    cjk = len(_CJK.findall(text))
    latin_words = len(_LATIN_WORD.findall(text))
    return "zh" if cjk > latin_words else "en"


def map_label(raw_label: str, lang: str) -> str:
    """模型原始标签 → 统一 schema；未知 → OTHER。"""
    table = _ZH_MAP if lang == "zh" else _EN_MAP
    return table.get(raw_label.strip().lower(), OTHER)


@dataclass
class MedicalNER:
    """中英分治 NER。后端可注入（测试/自定义）；缺省守卫加载真实模型。"""

    en_backend: Backend | None = None
    zh_backend: Backend | None = None

    def extract(self, text: str) -> list[Entity]:
        lang = detect_language(text)
        backend = self._backend(lang)
        return [
            Entity(text=t, label=map_label(rl, lang), start=s, end=e, lang=lang)
            for (t, rl, s, e) in backend(text)
        ]

    def _backend(self, lang: str) -> Backend:
        if lang == "zh":
            return self.zh_backend or self._load_zh()
        return self.en_backend or self._load_en()

    def _load_en(self) -> Backend:
        try:
            import spacy  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "英文医学 NER 需要 scispaCy/spaCy（en_ner_bc5cdr_md）；本地可注入 en_backend 测。"
            ) from exc
        nlp = spacy.load("en_ner_bc5cdr_md")
        return lambda text: [
            (e.text, e.label_, e.start_char, e.end_char) for e in nlp(text).ents
        ]

    def _load_zh(self) -> Backend:
        try:
            from transformers import pipeline  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "中文医学 NER 需要 transformers + CMeEE/CCKS BERT-NER 权重；本地可注入 zh_backend 测。"
            ) from exc
        ner = pipeline("token-classification", aggregation_strategy="simple")
        return lambda text: [
            (g["word"], g["entity_group"], int(g["start"]), int(g["end"])) for g in ner(text)
        ]


def as_entity_extractor(
    ner: MedicalNER, *, typed: bool = True
) -> Callable[[str], list]:
    """接到 [metrics_report] 的 `ner_fn`：typed=True 返回 (label, norm)，否则只返回 norm。"""
    def fn(text: str) -> list:
        ents = ner.extract(text)
        return [(e.label, e.norm) for e in ents] if typed else [e.norm for e in ents]

    return fn


__all__ = [
    "DRUG", "DISEASE", "ANATOMY", "FINDING", "PROCEDURE", "OTHER", "UNIFIED_LABELS",
    "RawSpan", "Backend", "Entity",
    "detect_language", "map_label", "MedicalNER", "as_entity_extractor",
]
