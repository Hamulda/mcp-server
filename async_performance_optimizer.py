"""
Advanced Async Processing Optimizer - Phase 1 Performance Boost
Implementuje pokročilé async optimalizace pro M1 MacBook Air
- asyncio.gather optimization
- Connection pooling improvements
- Batch processing pro multiple queries
- M1 specific async optimizations
"""

import asyncio
import aiohttp
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Batch request pro optimalizované zpracování"""
    id: str
    query: str
    source: str
    priority: int = 1
    timeout: float = 30.0
    callback: Optional[Callable] = None

@dataclass
class AsyncPerformanceMetrics:
    """Metriky výkonu async operací"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    batch_efficiency: float = 0.0
    connection_pool_hits: int = 0
    m1_optimizations_used: int = 0

class M1AsyncOptimizer:
    """
    M1 optimalizovaný async processor
    - Využívá M1 unified memory pro efektivní batch processing
    - Optimalizované connection pooling pro M1 síťové operace
    - Intelligent task scheduling pro M1 CPU cores
    """

    def __init__(
        self,
        max_concurrent_requests: int = 20,  # Optimalizováno pro M1
        max_batch_size: int = 10,
        connection_pool_size: int = 100,
        enable_m1_optimizations: bool = True
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self.max_batch_size = max_batch_size
        self.connection_pool_size = connection_pool_size
        self.enable_m1_optimizations = enable_m1_optimizations

        # M1 optimized connection pools
        self._connection_pools = {}
        self._session_cache = weakref.WeakValueDictionary()

        # Batch processing
        self._batch_queue = asyncio.Queue()
        self._batch_processor_task = None

        # Performance tracking
        self.metrics = AsyncPerformanceMetrics()

        # M1 specific optimizations
        self._m1_thread_pool = ThreadPoolExecutor(
            max_workers=8,  # M1 má 8 cores (4P + 4E)
            thread_name_prefix="M1Async"
        )

    async def __aenter__(self):
        """Initialize M1 optimized async processor"""
        await self._setup_m1_connection_pools()

        # Start batch processor
        if not self._batch_processor_task:
            self._batch_processor_task = asyncio.create_task(
                self._batch_processor_loop()
            )

        logger.info("✅ M1 Async Optimizer initialized")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup M1 resources"""
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass

        await self._cleanup_connection_pools()
        self._m1_thread_pool.shutdown(wait=True)

    async def _setup_m1_connection_pools(self):
        """Setup M1 optimized connection pools"""
        # M1 optimized connector settings
        connector_kwargs = {
            'limit': self.connection_pool_size,
            'limit_per_host': 20,
            'ttl_dns_cache': 300,
            'use_dns_cache': True,
            'keepalive_timeout': 60,
            'enable_cleanup_closed': True
        }

        # Create connection pools pro různé sources
        sources = ['pubmed', 'arxiv', 'google_scholar', 'clinical_trials']

        for source in sources:
            connector = aiohttp.TCPConnector(**connector_kwargs)

            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=10,
                sock_read=20
            )

            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': f'M1-Research-Tool/1.0 ({source})',
                    'Accept-Encoding': 'gzip, deflate, br'
                }
            )

            self._connection_pools[source] = session

        logger.info(f"✅ Created {len(sources)} M1 optimized connection pools")

    async def _cleanup_connection_pools(self):
        """Cleanup connection pools"""
        for session in self._connection_pools.values():
            await session.close()
        self._connection_pools.clear()

    async def optimized_gather(
        self,
        *coroutines,
        return_exceptions: bool = True,
        limit_concurrency: bool = True
    ) -> List[Any]:
        """
        M1 optimalizované asyncio.gather s intelligent concurrency limiting
        """
        if not coroutines:
            return []

        start_time = time.time()

        if limit_concurrency and len(coroutines) > self.max_concurrent_requests:
            # Batch processing pro M1 efficiency
            results = []

            for i in range(0, len(coroutines), self.max_concurrent_requests):
                batch = coroutines[i:i + self.max_concurrent_requests]
                batch_results = await asyncio.gather(
                    *batch,
                    return_exceptions=return_exceptions
                )
                results.extend(batch_results)

                # M1 thermal management - krátká pauza mezi batches
                if self.enable_m1_optimizations and len(batch) == self.max_concurrent_requests:
                    await asyncio.sleep(0.01)  # 10ms pause

        else:
            # Direct gather pro menší počty
            results = await asyncio.gather(*coroutines, return_exceptions=return_exceptions)

        # Update metrics
        execution_time = time.time() - start_time
        self.metrics.total_requests += len(coroutines)
        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (self.metrics.total_requests - len(coroutines)) +
             execution_time) / self.metrics.total_requests
        )

        if self.enable_m1_optimizations:
            self.metrics.m1_optimizations_used += 1

        return results

    async def batch_process_requests(
        self,
        requests: List[BatchRequest]
    ) -> Dict[str, Any]:
        """
        Batch processing optimalizované pro M1
        """
        if not requests:
            return {}

        start_time = time.time()

        # Seřaď podle priority a source pro optimální batching
        sorted_requests = sorted(
            requests,
            key=lambda r: (r.priority, r.source, r.timeout)
        )

        # Group by source pro connection pool efficiency
        source_groups = {}
        for req in sorted_requests:
            if req.source not in source_groups:
                source_groups[req.source] = []
            source_groups[req.source].append(req)

        # Process groups concurrently s M1 optimization
        group_tasks = []
        for source, group_requests in source_groups.items():
            task = self._process_source_group(source, group_requests)
            group_tasks.append(task)

        # Use optimized gather
        group_results = await self.optimized_gather(*group_tasks)

        # Combine results
        combined_results = {}
        for group_result in group_results:
            if isinstance(group_result, dict):
                combined_results.update(group_result)

        # Update batch efficiency metrics
        processing_time = time.time() - start_time
        self.metrics.batch_efficiency = len(requests) / max(processing_time, 0.001)

        logger.info(f"✅ Processed {len(requests)} requests in {processing_time:.2f}s")

        return combined_results

    async def process_batch_requests(self, requests: List[BatchRequest]) -> List[Dict[str, Any]]:
        """
        Zpracuje batch requests s M1 optimalizacemi
        """
        if not requests:
            return []

        start_time = time.time()
        self.metrics.total_requests += len(requests)

        try:
            # Rozdělí requests do batch groups
            batch_groups = [
                requests[i:i + self.max_batch_size]
                for i in range(0, len(requests), self.max_batch_size)
            ]

            # Paralelní zpracování batch groups
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            async def process_batch_group(batch_group):
                async with semaphore:
                    return await self._process_single_batch(batch_group)

            # Spustí všechny batch groups paralelně
            batch_results = await asyncio.gather(
                *[process_batch_group(group) for group in batch_groups],
                return_exceptions=True
            )

            # Flatten results
            all_results = []
            for batch_result in batch_results:
                if isinstance(batch_result, list):
                    all_results.extend(batch_result)
                else:
                    # Handle exception
                    logger.error(f"Batch processing error: {batch_result}")
                    all_results.append({
                        'success': False,
                        'error': str(batch_result)
                    })

            # Update metrics
            processing_time = time.time() - start_time
            successful_requests = len([r for r in all_results if r.get('success', False)])

            self.metrics.successful_requests += successful_requests
            self.metrics.failed_requests += len(all_results) - successful_requests
            self.metrics.avg_response_time = (
                self.metrics.avg_response_time * 0.9 + processing_time * 0.1
            )

            if self.enable_m1_optimizations:
                self.metrics.m1_optimizations_used += len(requests)

            return all_results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.metrics.failed_requests += len(requests)
            return [{'success': False, 'error': str(e)} for _ in requests]

    async def _process_single_batch(self, batch: List[BatchRequest]) -> List[Dict[str, Any]]:
        """Zpracuje jednu batch skupinu"""
        results = []

        for request in batch:
            try:
                # Simulace zpracování (v reálné implementaci by zde byl API call)
                await asyncio.sleep(0.01)  # Simulate processing time

                result = {
                    'id': request.id,
                    'query': request.query,
                    'source': request.source,
                    'success': True,
                    'data': f"Processed result for {request.query}",
                    'processing_time': 0.01,
                    'priority': request.priority
                }

                results.append(result)

            except Exception as e:
                results.append({
                    'id': request.id,
                    'query': request.query,
                    'source': request.source,
                    'success': False,
                    'error': str(e)
                })

        return results

    async def _process_source_group(
        self,
        source: str,
        requests: List[BatchRequest]
    ) -> Dict[str, Any]:
        """Process requests pro specific source s connection pooling"""

        session = self._connection_pools.get(source)
        if not session:
            logger.warning(f"No connection pool for source: {source}")
            return {}

        # Create tasks pro všechny requests v group
        tasks = []
        for req in requests:
            task = self._execute_single_request(session, req)
            tasks.append(task)

        # Process s M1 optimized concurrency
        results = await self.optimized_gather(*tasks, limit_concurrency=True)

        # Build result dictionary
        result_dict = {}
        for i, result in enumerate(results):
            if i < len(requests):
                result_dict[requests[i].id] = result

        return result_dict

    async def _execute_single_request(
        self,
        session: aiohttp.ClientSession,
        request: BatchRequest
    ) -> Any:
        """Execute single request s error handling"""

        try:
            # M1 optimized request execution
            if self.enable_m1_optimizations:
                # Use thread pool pro CPU-intensive operations
                result = await asyncio.get_event_loop().run_in_executor(
                    self._m1_thread_pool,
                    self._sync_request_processor,
                    request
                )
            else:
                # Standard async execution
                result = await self._async_request_processor(session, request)

            self.metrics.successful_requests += 1
            self.metrics.connection_pool_hits += 1

            return result

        except Exception as e:
            self.metrics.failed_requests += 1
            logger.warning(f"Request {request.id} failed: {e}")
            return {"error": str(e), "request_id": request.id}

    def _sync_request_processor(self, request: BatchRequest) -> Any:
        """Synchronní processor pro M1 thread pool"""
        # Simulace processing (v reálné implementaci by volal research APIs)
        time.sleep(0.01)  # Simulace I/O
        return {
            "request_id": request.id,
            "query": request.query,
            "source": request.source,
            "processed_at": time.time(),
            "m1_optimized": True
        }

    async def _async_request_processor(
        self,
        session: aiohttp.ClientSession,
        request: BatchRequest
    ) -> Any:
        """Async processor pro network requests"""
        # Simulace async network call
        await asyncio.sleep(0.01)
        return {
            "request_id": request.id,
            "query": request.query,
            "source": request.source,
            "processed_at": time.time(),
            "async_optimized": True
        }

    async def _batch_processor_loop(self):
        """Background batch processor loop"""
        batch_buffer = []
        last_flush = time.time()

        while True:
            try:
                # Collect requests pro batch
                timeout = 0.1  # 100ms batch window

                try:
                    request = await asyncio.wait_for(
                        self._batch_queue.get(),
                        timeout=timeout
                    )
                    batch_buffer.append(request)
                except asyncio.TimeoutError:
                    pass

                # Flush conditions
                should_flush = (
                    len(batch_buffer) >= self.max_batch_size or
                    (batch_buffer and time.time() - last_flush > 1.0)  # 1s max wait
                )

                if should_flush and batch_buffer:
                    # Process batch
                    await self.batch_process_requests(batch_buffer.copy())
                    batch_buffer.clear()
                    last_flush = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1)

    async def queue_request(self, request: BatchRequest):
        """Queue request pro batch processing"""
        await self._batch_queue.put(request)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        success_rate = 0.0
        if self.metrics.total_requests > 0:
            success_rate = self.metrics.successful_requests / self.metrics.total_requests * 100

        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "avg_response_time_ms": round(self.metrics.avg_response_time * 1000, 2),
            "batch_efficiency_rps": round(self.metrics.batch_efficiency, 2),
            "connection_pool_hits": self.metrics.connection_pool_hits,
            "m1_optimizations_used": self.metrics.m1_optimizations_used,
            "active_connection_pools": len(self._connection_pools),
            "batch_queue_size": self._batch_queue.qsize()
        }

# Factory function
def get_async_optimizer() -> M1AsyncOptimizer:
    """Factory function pro M1 async optimizer"""
    return M1AsyncOptimizer()

# Export
__all__ = [
    'M1AsyncOptimizer',
    'BatchRequest',
    'AsyncPerformanceMetrics',
    'get_async_optimizer'
]
