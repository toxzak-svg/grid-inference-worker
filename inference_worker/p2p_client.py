# SPDX-FileCopyrightText: 2026 AI Power Grid
# SPDX-License-Identifier: AGPL-3.0-or-later

"""P2P client for decentralized Grid workers.

Replaces WebSocket connection with libp2p gossipsub. The worker:
1. Joins the P2P mesh
2. Subscribes to job topics for its models
3. Claims jobs using deterministic selection
4. Streams results back via gossipsub

Enable with P2P_ENABLED=true in your .env.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx

from .config import Settings

logger = logging.getLogger(__name__)

# Topic structure
TOPIC_PREFIX = "/aipg/1"


def job_topic(model: str) -> str:
    """Get gossipsub topic for jobs targeting a model."""
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"{TOPIC_PREFIX}/jobs/{safe_model}"


def claims_topic() -> str:
    """Get the global claims topic."""
    return f"{TOPIC_PREFIX}/claims"


def results_topic(job_id: str) -> str:
    """Get topic for a specific job's results."""
    return f"{TOPIC_PREFIX}/results/{job_id}"


def _strip_thinking_tags(text: str) -> str:
    """Remove think-tag blocks from reasoning models."""
    import re
    if not text:
        return text
    return re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL).strip()


@dataclass
class JobRequest:
    """A job received from the P2P network."""
    id: str
    model: str
    payload: dict[str, Any]
    max_cost: int
    user_pubkey: str
    signature: str
    timestamp: float
    ttl: int = 60

    @classmethod
    def from_json(cls, data: str) -> "JobRequest":
        d = json.loads(data)
        return cls(**d)

    def is_expired(self) -> bool:
        return time.time() > self.timestamp + self.ttl

    def seed(self) -> bytes:
        """Get random seed for claim resolution."""
        return bytes.fromhex(self.signature[:64])


@dataclass
class JobClaim:
    """A claim broadcast to prevent double-processing."""
    job_id: str
    worker_id: str
    worker_pubkey: str
    price: int
    signature: str
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, data: str) -> "JobClaim":
        return cls(**json.loads(data))


def compute_claim_score(job_id: str, seed: bytes, worker_id: str) -> bytes:
    """Compute deterministic score for claim resolution. Lower wins."""
    data = job_id.encode() + seed + worker_id.encode()
    return hashlib.sha256(data).digest()


def should_claim(job: JobRequest, my_worker_id: str, known_workers: list[str]) -> bool:
    """Determine if this worker should claim the job."""
    if not known_workers:
        return True

    seed = job.seed()
    my_score = compute_claim_score(job.id, seed, my_worker_id)

    for worker_id in known_workers:
        if worker_id == my_worker_id:
            continue
        their_score = compute_claim_score(job.id, seed, worker_id)
        if their_score < my_score:
            return False

    return True


class P2PWorker:
    """P2P-based worker using libp2p gossipsub."""

    def __init__(self):
        Settings.validate()
        self.model_name: str = Settings.MODEL_NAME
        self.grid_model_name: str = Settings.GRID_MODEL_NAME or self._build_grid_model_name()
        self.backend = httpx.AsyncClient(timeout=120)

        # P2P state
        self._host = None
        self._pubsub = None
        self.peer_id: str = ""
        self._known_workers: set[str] = set()
        self._claimed_jobs: dict[str, JobClaim] = {}
        self._running = False

        # Stats
        self._jobs_completed = 0
        self._total_den = 0.0

    def _build_grid_model_name(self) -> str:
        if Settings.BACKEND_TYPE == "ollama":
            return f"grid/{self.model_name}"
        return f"grid/{self.model_name}"

    def _get_completions_url(self) -> str:
        if Settings.BACKEND_TYPE == "ollama":
            return f"{Settings.OLLAMA_URL}/v1/chat/completions"
        return f"{Settings.OPENAI_URL}/chat/completions"

    def _get_auth_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if Settings.BACKEND_TYPE != "ollama" and Settings.OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {Settings.OPENAI_API_KEY}"
        return headers

    async def start(self):
        """Start the P2P node and join the mesh."""
        if self._running:
            return

        try:
            from libp2p import new_host
            from libp2p.pubsub.gossipsub import GossipSub
            from libp2p.pubsub.pubsub import Pubsub
            from libp2p.peer.peerinfo import info_from_p2p_addr
            import multiaddr
        except ImportError as e:
            logger.error(f"libp2p not installed: {e}")
            logger.error("Install with: pip install libp2p trio")
            raise

        listen_port = Settings.P2P_LISTEN_PORT
        listen_addr = f"/ip4/0.0.0.0/tcp/{listen_port}"

        logger.info(f"Starting P2P node on {listen_addr}...")

        # Create host
        self._host = new_host()

        # Initialize gossipsub
        gs = GossipSub(
            degree=6,
            degree_low=4,
            degree_high=12,
            time_to_live=5,
        )
        self._pubsub = Pubsub(self._host, gs)

        self.peer_id = self._host.get_id().to_string()
        self._known_workers.add(self.peer_id)

        # Connect to bootstrap peers
        bootstrap_peers = Settings.P2P_BOOTSTRAP_PEERS
        for peer_addr in bootstrap_peers:
            try:
                maddr = multiaddr.Multiaddr(peer_addr)
                info = info_from_p2p_addr(maddr)
                await self._host.connect(info)
                logger.info(f"Connected to bootstrap peer: {info.peer_id}")
            except Exception as e:
                logger.warning(f"Failed to connect to {peer_addr}: {e}")

        self._running = True
        logger.info(f"P2P node started. Peer ID: {self.peer_id}")

    async def stop(self):
        """Stop the P2P node."""
        self._running = False
        await self.backend.aclose()
        logger.info("P2P node stopped")

    async def run(self):
        """Main worker loop."""
        await self.start()

        logger.info(f"🚀 P2P Worker starting | model={self.grid_model_name}")
        logger.info(f"📡 Backend: {Settings.BACKEND_TYPE} @ {self._get_completions_url()}")
        logger.info(f"🔗 Peer ID: {self.peer_id}")

        # Subscribe to job topic
        topic = job_topic(self.grid_model_name)
        logger.info(f"📥 Subscribing to {topic}")

        await self._pubsub.subscribe(topic)
        await self._pubsub.subscribe(claims_topic())

        # Process messages
        try:
            while self._running:
                # In a real implementation, we'd use trio's async for
                # For now, poll subscriptions
                await self._process_messages(topic)
                await asyncio.sleep(0.1)
        finally:
            await self.stop()

    async def _process_messages(self, job_topic_name: str):
        """Process incoming messages from subscriptions."""
        # This is a simplified version - real impl would use trio properly
        # For demonstration, we'll check for messages
        pass  # In real impl, async iterate over subscription

    async def _handle_job(self, job: JobRequest) -> None:
        """Process a single job."""
        job_id = job.id
        payload = job.payload

        # Check if we should claim
        if not should_claim(job, self.peer_id, list(self._known_workers)):
            logger.debug(f"Not our turn for job {job_id[:8]}")
            return

        # Check if already claimed by someone else
        if job_id in self._claimed_jobs:
            claim = self._claimed_jobs[job_id]
            if claim.worker_id != self.peer_id:
                logger.debug(f"Job {job_id[:8]} already claimed by {claim.worker_id[:8]}")
                return

        # Broadcast our claim
        claim = JobClaim(
            job_id=job_id,
            worker_id=self.peer_id,
            worker_pubkey="",
            price=0,
            signature="",
        )
        self._claimed_jobs[job_id] = claim
        await self._pubsub.publish(claims_topic(), claim.to_json().encode())
        logger.info(f"📋 Claimed job {job_id[:8]}")

        # Extract prompt
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_length", 512))
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.9))

        logger.info(f"📥 Processing {job_id[:8]} | max_tokens={max_tokens}")

        # Build OpenAI-compatible request
        openai_payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        url = self._get_completions_url()
        headers = self._get_auth_headers()
        result_topic = results_topic(job_id)

        full_text = ""
        token_count = 0
        start_time = time.time()

        try:
            async with self.backend.stream("POST", url, json=openai_payload, headers=headers) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error(f"Backend error {response.status_code}: {body[:200]}")
                    await self._pubsub.publish(
                        result_topic,
                        json.dumps({
                            "job_id": job_id,
                            "worker_id": self.peer_id,
                            "type": "error",
                            "error": {"message": f"Backend error: {response.status_code}", "code": response.status_code},
                        }).encode()
                    )
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            content = _strip_thinking_tags(content) if "<think" in content else content
                            if content:
                                full_text += content
                                token_count += 1

                                # Publish token to result topic
                                await self._pubsub.publish(
                                    result_topic,
                                    json.dumps({
                                        "job_id": job_id,
                                        "worker_id": self.peer_id,
                                        "type": "token",
                                        "token": {"text": content, "index": token_count},
                                    }).encode()
                                )
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            logger.error(f"Backend error: {e}")
            await self._pubsub.publish(
                result_topic,
                json.dumps({
                    "job_id": job_id,
                    "worker_id": self.peer_id,
                    "type": "error",
                    "error": {"message": str(e), "code": 0},
                }).encode()
            )
            return

        # Publish completion
        gen_time = time.time() - start_time
        await self._pubsub.publish(
            result_topic,
            json.dumps({
                "job_id": job_id,
                "worker_id": self.peer_id,
                "type": "done",
                "done": {
                    "full_text": full_text,
                    "token_count": token_count,
                    "receipt_signature": "",
                },
            }).encode()
        )

        self._jobs_completed += 1
        tps = token_count / gen_time if gen_time > 0 else 0
        logger.info(
            f"✅ {job_id[:8]} | {token_count} tokens | {gen_time:.1f}s | "
            f"{tps:.1f} TPS | total jobs: {self._jobs_completed}"
        )

    async def _handle_claim(self, claim: JobClaim) -> None:
        """Handle incoming claim from another worker."""
        existing = self._claimed_jobs.get(claim.job_id)
        if not existing or claim.timestamp < existing.timestamp:
            self._claimed_jobs[claim.job_id] = claim
            self._known_workers.add(claim.worker_id)
            logger.debug(f"Recorded claim: {claim.worker_id[:8]} -> {claim.job_id[:8]}")


async def run_p2p_worker():
    """Entry point for P2P worker mode."""
    worker = P2PWorker()
    await worker.run()
