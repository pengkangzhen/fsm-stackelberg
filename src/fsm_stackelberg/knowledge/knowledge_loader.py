"""知识加载器模块

支持渐进式知识披露：
- 知识文件带 YAML front matter（name, description）
- 生成知识目录供建模专家参考
- 生成知识内容作为附录
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterable

import yaml


@dataclass
class KnowledgeModule:
    """知识模块"""
    name: str
    description: str
    content: str
    file_path: str


@dataclass
class AssembledKnowledge:
    """组装后的知识对象"""
    knowledge_catalog: str       # 知识目录表格
    knowledge_content: str       # 知识内容（作为附录）
    knowledge_modules: List[str] = field(default_factory=list)


class KnowledgeLoader:
    """渐进式知识加载器

    扫描知识文件目录，解析 front matter，生成知识目录和内容。
    """

    def __init__(self, knowledge_base_path: Optional[str] = None):
        if knowledge_base_path is None:
            # 默认路径
            knowledge_base_path = Path(__file__).parent
        self.knowledge_base_path = Path(knowledge_base_path)
        self.modules: List[KnowledgeModule] = []
        self.logger = logging.getLogger(__name__)
        self._scan_knowledge_modules()

    def _scan_knowledge_modules(self) -> None:
        """扫描所有知识文件，提取元数据和内容"""
        domains_path = self.knowledge_base_path / "domains"
        if not domains_path.exists():
            self.logger.warning(f"知识目录不存在: {domains_path}")
            return

        for md_file in domains_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                # 解析 YAML front matter
                description = ""
                name = md_file.stem  # 默认使用文件名
                body = content

                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        front_matter = parts[1]
                        body = parts[2].strip()
                        metadata = yaml.safe_load(front_matter)
                        description = metadata.get("description", "")
                        # 优先使用front matter中的name字段
                        name = metadata.get("name", md_file.stem)

                self.modules.append(KnowledgeModule(
                    name=name,
                    description=description,
                    content=body,
                    file_path=str(md_file.relative_to(self.knowledge_base_path))
                ))
                self.logger.debug(f"加载知识模块: {name}")

            except Exception as e:
                self.logger.warning(f"解析知识文件失败 {md_file}: {e}")

    @staticmethod
    def _normalize_excluded_modules(excluded_modules: Optional[Iterable[str]] = None) -> set[str]:
        return {name for name in (excluded_modules or []) if name}

    def list_module_names(self, excluded_modules: Optional[Iterable[str]] = None) -> List[str]:
        """Return available module names after exclusions."""
        excluded = self._normalize_excluded_modules(excluded_modules)
        return [module.name for module in self.modules if module.name not in excluded]

    def get_knowledge_catalog(self, excluded_modules: Optional[Iterable[str]] = None) -> str:
        """生成知识目录表格（供建模专家参考）"""
        excluded = self._normalize_excluded_modules(excluded_modules)
        visible_modules = [module for module in self.modules if module.name not in excluded]
        if not visible_modules:
            return "暂无可用知识"

        lines = ["| 文件名 | 描述 |", "|--------|------|"]
        for m in visible_modules:
            lines.append(f"| {m.name} | {m.description} |")
        return "\n".join(lines)

    def get_knowledge_catalog_only(self, excluded_modules: Optional[Iterable[str]] = None) -> str:
        """仅返回知识目录（不含内容）- 用于渐进式加载的第一阶段"""
        excluded = self._normalize_excluded_modules(excluded_modules)
        visible_modules = [module for module in self.modules if module.name not in excluded]
        if not visible_modules:
            return "暂无可用知识"

        lines = ["| 模块名 | 描述 |", "|--------|------|"]
        for m in visible_modules:
            lines.append(f"| {m.name} | {m.description} |")
        return "\n".join(lines)

    def get_all_knowledge_content(self, excluded_modules: Optional[Iterable[str]] = None) -> str:
        """生成所有知识的详细内容（作为附录）"""
        excluded = self._normalize_excluded_modules(excluded_modules)
        visible_modules = [module for module in self.modules if module.name not in excluded]
        if not visible_modules:
            return ""

        parts = []
        for m in visible_modules:
            parts.append(f"### {m.name}\n\n{m.content}")
        return "\n\n".join(parts)

    def get_full_knowledge_prompt(self, excluded_modules: Optional[Iterable[str]] = None) -> str:
        """生成完整的知识提示（目录 + 附录）"""
        catalog = self.get_knowledge_catalog(excluded_modules=excluded_modules)
        content = self.get_all_knowledge_content(excluded_modules=excluded_modules)

        if not content:
            return ""

        return f"""## 可选的领域知识（如需要可参考）

以下是可用的领域知识文件摘要，完整内容见附录：

{catalog}

---

## 附录：领域知识详细内容

{content}"""

    def get_knowledge_module(
        self,
        module_name: str,
        excluded_modules: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """根据模块名加载单个知识模块的内容"""
        excluded = self._normalize_excluded_modules(excluded_modules)
        if module_name in excluded:
            self.logger.info(f"知识模块被实验配置排除: {module_name}")
            return None
        for module in self.modules:
            if module.name == module_name:
                return f"### {module.name}\n\n{module.content}"
        self.logger.warning(f"未找到知识模块: {module_name}")
        return None

    def get_knowledge_by_names(
        self,
        module_names: List[str],
        excluded_modules: Optional[Iterable[str]] = None,
    ) -> str:
        """根据模块名列表加载多个知识模块"""
        if not module_names:
            return ""
        
        parts = []
        for name in module_names:
            content = self.get_knowledge_module(name, excluded_modules=excluded_modules)
            if content:
                parts.append(content)
        
        return "\n\n---\n\n".join(parts) if parts else ""

    def search_modules_by_keywords(self, keywords: List[str]) -> List[str]:
        """根据关键词搜索相关模块名称"""
        if not keywords:
            return []
        
        matched_modules = []
        for module in self.modules:
            # 在名称、描述和内容中搜索关键词
            search_text = f"{module.name} {module.description} {module.content}".lower()
            
            for keyword in keywords:
                if keyword.lower() in search_text:
                    matched_modules.append(module.name)
                    break  # 避免重复添加同一个模块
        
        return matched_modules

    def recommend_modules(
        self,
        problem_description: str,
        excluded_modules: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """根据问题描述智能推荐相关模块"""
        if not problem_description:
            return []
        excluded = self._normalize_excluded_modules(excluded_modules)
        
        # 关键词映射规则
        keyword_mapping = {
            "two-stage": ["tslp-two-stage-structure"],
            "two stage": ["tslp-two-stage-structure"],
            "两阶段": ["tslp-two-stage-structure"],
            "stochastic": ["tslp-two-stage-structure", "tslp-demand-scenarios"],
            "随机": ["tslp-two-stage-structure", "tslp-demand-scenarios"],
            "scenario": ["tslp-demand-scenarios"],
            "情景": ["tslp-demand-scenarios"],
            "eta": ["tslp-demand-scenarios"],
            "demand": ["tslp-demand-scenarios"],
            "需求": ["tslp-demand-scenarios"],
            "first stage": ["tslp-stage1-sea"],
            "第一阶段": ["tslp-stage1-sea"],
            "y_in": ["tslp-stage1-sea"],
            "y_out": ["tslp-stage1-sea"],
            "hub": ["tslp-stage1-sea", "tslp-network-nodes"],
            "枢纽": ["tslp-stage1-sea", "tslp-network-nodes"],
            "second stage": ["tslp-stage2-inland"],
            "第二阶段": ["tslp-stage2-inland"],
            "inland": ["tslp-stage2-inland"],
            "内陆": ["tslp-stage2-inland"],
            "inventory": ["tslp-stage2-inland"],
            "库存": ["tslp-stage2-inland"],
            "lease": ["tslp-stage2-inland"],
            "租箱": ["tslp-stage2-inland"],
            "spill": ["tslp-stage2-inland"],
            "xi": ["tslp-exogenous-supply"],
            "exogenous": ["tslp-exogenous-supply"],
            "外生": ["tslp-exogenous-supply"],
            "supply": ["tslp-exogenous-supply"],
            "dry": ["tslp-network-nodes"],
            "陆港": ["tslp-network-nodes"],
            "spoke": ["tslp-network-nodes"],
            "海港": ["tslp-network-nodes"],
            "sample.json": ["tslp-data-access"],
            "data file": ["tslp-data-access"],
            "数据": ["tslp-data-access"],
            "gurobi": ["tslp-data-access", "tslp-two-stage-structure"],
            "sets": ["tslp-data-access"],
            "parameters": ["tslp-data-access"],
        }
        
        # 从问题描述中提取关键词
        desc_lower = problem_description.lower()
        recommended = set()
        
        for keyword, modules in keyword_mapping.items():
            if keyword in desc_lower:
                recommended.update(modules)
        
        # 如果没有匹配到任何模块，返回所有模块作为默认推荐
        if not recommended:
            return self.list_module_names(excluded_modules=excluded)
        
        return [name for name in recommended if name not in excluded]
