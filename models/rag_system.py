# models/rag_system.py
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple

class RAGSystem:
    def __init__(self):
        self.knowledge_bases: Dict[str, Dict[str, Any]] = {}
        self.load_knowledge_bases()

    # ---------- KB 加载 ----------

    def load_knowledge_bases(self) -> None:
        """Load all knowledge base files from data/knowledge_base"""
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")
        )

        kb_files = [
            "healthy_neutral.json",
            "healthy_toxic.json",
            "anxiety_neutral.json",
            "anxiety_toxic.json",
            "depression_neutral.json",
            "depression_toxic.json",
        ]

        for filename in kb_files:
            twin_type, scenario = filename.replace(".json", "").split("_", 1)
            key = f"{twin_type}_{scenario}"
            file_path = os.path.join(data_dir, filename)

            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self.knowledge_bases[key] = json.load(f)
                    print(f"[RAG] Loaded knowledge base: {key}")
                except Exception as e:
                    print(f"[RAG] Error loading {filename}: {e}")
                    self.knowledge_bases[key] = self._create_placeholder_kb(twin_type, scenario)
            else:
                self.knowledge_bases[key] = self._create_placeholder_kb(twin_type, scenario)
                print(f"[RAG] Created placeholder knowledge base: {key}")

    def _create_placeholder_kb(self, twin_type: str, scenario: str) -> Dict[str, Any]:
        """Create placeholder knowledge base structure"""
        return {
            "metadata": {
                "twin_type": twin_type,
                "scenario": scenario,
                "description": f"Knowledge base for {twin_type} twin in {scenario} environment",
            },
            "memories": [
                {
                    "id": f"{twin_type}_{scenario}_placeholder",
                    "content": f"This is a placeholder memory for {twin_type} in {scenario} context.",
                    "keywords": [twin_type, scenario, "placeholder"],
                    "emotional_context": "neutral",
                    "relevance_score": 0.1,
                    "source": "placeholder",
                }
            ],
        }

    # ---------- 关键词与打分 ----------

    def _extract_keywords(self, text: str) -> List[str]:
        """Very simple tokenizer/keyword extractor"""
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        # 可按需去停用词
        stop = {"the", "a", "an", "and", "or", "is", "are", "to", "of", "in", "on", "for", "with"}
        return [w for w in words if w not in stop]

    def _calculate_relevance_score(self, query_words: List[str], memory: Dict[str, Any]) -> float:
        """Keyword overlap + optional keyword field boost"""
        text = (memory.get("content") or "").lower()
        mem_words = set(self._extract_keywords(text))
        if not mem_words:
            return 0.0

        overlap = len(set(query_words) & mem_words)

        # 额外利用 memory["keywords"] 字段
        kw_list = memory.get("keywords") or []
        kw_overlap = len(set(query_words) & set([str(k).lower() for k in kw_list]))

        # 简单线性组合（可按需调整权重）
        score = overlap + 0.5 * kw_overlap

        # 如果有作者给的初始 relevance_score，可做一个温和加成
        base = float(memory.get("relevance_score", 0.0))
        score += 0.2 * base
        return score

    # ---------- 检索主流程 ----------

    def retrieve_relevant_info(
        self,
        query: str,
        twin_type: str,
        scenario: str,
        top_k: int = 3,
        tag_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        返回结构化结果，示例：
        {
          "memories": [ {memory1}, {memory2}, ... ],
          "context_text": "...用于拼接到提示的文本...",
          "meta": {"key": "healthy_neutral", "top_k": 3}
        }
        """
        key = f"{twin_type}_{scenario}"
        try:
            kb = self.knowledge_bases.get(key)
            if not kb:
                return {"memories": [], "context_text": "", "meta": {"key": key, "error": "kb_not_found"}}

            memories = kb.get("memories", [])
            if not memories:
                return {"memories": [], "context_text": "", "meta": {"key": key, "error": "empty_kb"}}

            # 过滤标签（如果你的 memory 里有 tags 字段）
            if tag_filter:
                filtered = []
                for m in memories:
                    tags = [t.lower() for t in (m.get("tags") or [])]
                    if any(t.lower() in tags for t in tag_filter):
                        filtered.append(m)
                memories = filtered

            query_words = self._extract_keywords((query or "").lower())
            scored: List[Tuple[Dict[str, Any], float]] = []

            for memory in memories:
                score = self._calculate_relevance_score(query_words, memory)
                if score > 0:
                    scored.append((memory, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            top = [m for (m, _) in scored[:max(1, top_k)]]

            # 组装可直接拼到 prompt 的上下文
            context_lines = []
            for i, m in enumerate(top, 1):
                src = m.get("source") or ""
                context_lines.append(f"[KB#{i}] {m.get('content','').strip()}  {('('+src+')') if src else ''}".strip())
            context_text = "\n".join(context_lines)

            return {
                "memories": top,
                "context_text": context_text,
                "meta": {"key": key, "top_k": top_k, "count_all": len(memories)},
            }

        except Exception as e:
            # 防御式返回，避免导入阶段直接崩溃
            return {"memories": [], "context_text": "", "meta": {"key": key, "error": str(e)}}
