 
import asyncio
import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb

from knowledge.embedding import LocalHashEmbeddingFunction
from core.tenant import DEFAULT_TENANT_ID, normalize_tenant_id

logger = logging.getLogger(__name__)

__all__ = ["KnowledgeBase"]


class KnowledgeBase:

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        allow_local_fallback: Optional[bool] = None,
    ):
                                                
        self._use_server = False
                                                               
                                     
        self._embedding_function = LocalHashEmbeddingFunction()
        if allow_local_fallback is None:
            allow_local_fallback = os.getenv("ECHOMIND_ALLOW_LOCAL_CHROMA_FALLBACK", "false").lower() in {
                "1", "true", "yes", "on"
            }
        try:
                                                                           
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            if not allow_local_fallback:
                raise RuntimeError(
                    "ChromaDB 服务不可用，且未允许本地降级；"
                    "开发环境请设置 ECHOMIND_ALLOW_LOCAL_CHROMA_FALLBACK=true"
                ) from None
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._collection_name = "knowledge_base_local_v2"
        else:
            self._collection_name = "knowledge_base_v2"

                                                  
                                                
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "EchoMind RAG 知识库"},
            embedding_function=self._embedding_function,
        )

        self._migrate_legacy_metadata()

                        
        if self._collection.count() == 0:
            self._load_default_docs()

                                                                            

    def add_documents(self, documents: List[Dict[str, str]], tenant_id: str = DEFAULT_TENANT_ID) -> int:
        
        tenant_id = normalize_tenant_id(tenant_id)
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.sha256(f"{title}\0{i}\0{chunk}".encode("utf-8")).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "tenant_id": tenant_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })

        if ids:
            writer = getattr(self._collection, "upsert", None) or self._collection.add
            writer(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(
        self, documents: List[Dict[str, str]], tenant_id: str = DEFAULT_TENANT_ID
    ) -> int:
                                                 
        return await asyncio.to_thread(self.add_documents, documents, tenant_id)

    def search(
        self, query: str, top_k: int = 5, tenant_id: str = DEFAULT_TENANT_ID
    ) -> List[Dict[str, Any]]:
           
        query = self._validate_query(query)
        top_k = self._validate_top_k(top_k)
        tenant_id = normalize_tenant_id(tenant_id)
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"tenant_id": tenant_id},
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),                       
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

    async def search_async(
        self, query: str, top_k: int = 5, tenant_id: str = DEFAULT_TENANT_ID
    ) -> List[Dict[str, Any]]:
                                               
        return await asyncio.to_thread(self.search, query, top_k, tenant_id)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self, tenant_id: Optional[str] = None) -> int:
                         
        return await asyncio.to_thread(self._count_for_tenant, tenant_id)

                                                                             

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
           
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        tenant_id = DEFAULT_TENANT_ID
        if isinstance(context, dict):
            tenant_id = context.get("tenant_id", DEFAULT_TENANT_ID)
        return await self.search_async(query, top_k=top_k, tenant_id=tenant_id)

    @staticmethod
    def _validate_query(query: Any) -> str:
        value = str(query or "").strip()
        if not value:
            raise ValueError("知识库查询不能为空")
        if len(value) > 2000:
            raise ValueError("知识库查询长度不能超过 2000 个字符")
        return value

    @staticmethod
    def _validate_top_k(top_k: Any) -> int:
        if isinstance(top_k, bool):
            raise ValueError("top_k 必须是正整数")
        try:
            value = int(top_k)
        except (TypeError, ValueError) as ex:
            raise ValueError("top_k 必须是正整数") from ex
        if not 1 <= value <= 50:
            raise ValueError("top_k 必须在 1 到 50 之间")
        return value

    async def readiness(self) -> Dict[str, Any]:
        return {
            "ready": True,
            "mode": "server" if self._use_server else "local_hash_embedding",
            "collection": self._collection_name,
            "documents": await self.doc_count_async(),
        }

                                                                            

    def _count_for_tenant(self, tenant_id: Optional[str] = None) -> int:
        if tenant_id is None:
            return self._collection.count()
        tenant_id = normalize_tenant_id(tenant_id)
        result = self._collection.get(where={"tenant_id": tenant_id}, include=[])
        return len(result.get("ids", []))

    def _migrate_legacy_metadata(self) -> None:
                                                    
        if not hasattr(self._collection, "get"):
                                                  
            return
        try:
            result = self._collection.get(include=["metadatas"])
            ids = result.get("ids", [])
            metadatas = result.get("metadatas", [])
            updates = []
            update_ids = []
            for doc_id, metadata in zip(ids, metadatas):
                metadata = dict(metadata or {})
                if not metadata.get("tenant_id"):
                    metadata["tenant_id"] = DEFAULT_TENANT_ID
                    update_ids.append(doc_id)
                    updates.append(metadata)
            if update_ids and hasattr(self._collection, "update"):
                self._collection.update(ids=update_ids, metadatas=updates)
                logger.info("已为 %s 个旧知识片段补齐 default 租户元数据", len(update_ids))
        except Exception as ex:
            raise RuntimeError("知识库租户元数据迁移失败") from ex

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
                                                    
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
               
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
                                  
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs, tenant_id=DEFAULT_TENANT_ID)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
