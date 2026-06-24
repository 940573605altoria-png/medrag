"""RAG 单测（T065 起步）—— 先覆盖 T013 向量库。

集合名校验/缺包报错本机即可跑；add/query/where 功能测需 chromadb（本机装了则真跑，
否则 skip，AutoDL 上覆盖）。用内存 EphemeralClient，互不污染。
"""

from __future__ import annotations

import pytest

from src.rag import store


def test_collections_constant():
    assert store.COLLECTIONS == ("a_drug", "b_medqa", "c_text", "c_img_whole", "c_img_roi")


def test_validate_collection_rejects_unknown():
    assert store.validate_collection("c_text") == "c_text"
    with pytest.raises(ValueError, match="未知 collection"):
        store.validate_collection("bogus")


def test_missing_chromadb_clear_error():
    try:
        import chromadb  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="chromadb"):
            store.VectorStore()
    else:
        pytest.skip("本环境已装 chromadb，跳过缺依赖分支")


# ── 功能测（需 chromadb，内存库）────────────────────────────────────

def _store():
    pytest.importorskip("chromadb")
    vs = store.VectorStore()                         # path=None → 内存
    # chromadb EphemeralClient 进程内共享 → 先 reset 保证测试隔离（不受其他测试遗留维度影响）
    for c in store.COLLECTIONS:
        vs.reset(c)
    return vs


def test_add_and_count():
    vs = _store()
    vs.add("a_drug", ids=["d1", "d2"], embeddings=[[1.0, 0.0], [0.0, 1.0]],
           documents=["drug one", "drug two"])
    assert vs.count("a_drug") == 2


def test_query_returns_nearest():
    vs = _store()
    vs.add("c_text", ids=["x", "y", "z"],
           embeddings=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    res = vs.query("c_text", [[0.9, 0.1]], n_results=1)
    assert res["ids"][0][0] == "x"                   # 最近的是 x


def test_where_metadata_filter():
    vs = _store()
    vs.add("c_img_roi", ids=["s1", "m1"],
           embeddings=[[1.0, 0.0], [0.9, 0.1]],
           metadatas=[{"band": "small"}, {"band": "medium"}])
    res = vs.query("c_img_roi", [[1.0, 0.0]], n_results=5, where={"band": "small"})
    assert res["ids"][0] == ["s1"]                   # 仅返回 small


def test_add_unknown_collection_raises():
    vs = _store()
    with pytest.raises(ValueError, match="未知 collection"):
        vs.add("nope", ids=["a"], embeddings=[[1.0, 0.0]])
