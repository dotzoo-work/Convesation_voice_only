import asyncio
import json
import logging
import time
import aiohttp
import os
from livekit.plugins import openai
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env.local'))

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
    function_tool,
    RunContext,
)
from livekit.plugins import (
    noise_cancellation,
    silero,
    deepgram,
    sarvam,
    cartesia,
    openai,
)

logger = logging.getLogger("agent")

DEEPGRAM_TTS_MAP = {
    "en": "aura-2-thalia-en",
    "de": "aura-2-kara-de",
    "fr": "aura-2-agathe-fr",
    "es": "aura-2-carina-es",
}


MINIMAL_HARD_RULES = [
    "prime minister",
    "cricket score",
    "weather today",
]


def normalize_lang(lang: str) -> str:
    mapping = {
        "hi": "hi", "hindi": "hi",
        "pa": "pa", "punjabi": "pa",
        "en": "en", "english": "en",
        "es": "es", "spanish": "es",
        "de": "de", "german": "de",
        "fr": "fr", "french": "fr",
    }
    return mapping.get(lang.lower(), "en")


def select_stt(lang: str):
    if lang in ["hi", "pa"]:
        logger.info("USING SARVAM STT (saaras:v3)")
        return sarvam.STT(language=f"{lang}-IN", model="saaras:v3")
    logger.info("USING DEEPGRAM STT")
    return deepgram.STT(model="nova-3", language=lang)


def select_tts(lang: str):
    # Hindi / Punjabi keep Sarvam
    if lang == "hi":
        return sarvam.TTS(
            target_language_code="hi-IN",
            model="bulbul:v2",
            speaker="anushka",
            pace=0.85
        )
    if lang == "pa":
        return sarvam.TTS(
            target_language_code="pa-IN",
            model="bulbul:v2",
            speaker="vidya",
            pace=0.85
        )
    # English + others → Cartesia
    return cartesia.TTS(
        model="sonic-2",
        voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        language=lang,
    )


def fast_classify(q: str):
    q = q.lower()
    if any(x in q for x in ["pain", "bleeding", "swelling"]):
        return "emergency"
    if any(x in q for x in ["appointment", "book"]):
        return "appointment"
    if any(x in q for x in ["open", "time", "hours"]):
        return "clinic_hours"
    return None


def clean_text(text: str) -> str:
    text = text.replace("*", "").replace("#", "").replace("\n", " ")
    text = text.replace("Dr.", "Doctor").replace("DDS", "")
    return text.strip()[:250]


class Assistant(Agent):
    def __init__(self, lang: str = "en", qdrant_client=None, openai_client=None) -> None:
        self._lang = lang
        super().__init__(
            instructions=f"""You are a dental  vitual clinic assistant for Edmonds Bay Dental.
            CLINIC INFORMATION:
- Locations: 
  primary: Edmonds Bay Dental (51 W Dayton Street Suite 301, Edmonds, WA 98020, Phone: 425-775-5162).
  Secondary: Pacific Highway Dental (27020 Pacific Highway South, Suite C, Kent, WA 98032, Phone: 253-529-9434)
 Language: 
     Doctor: English, Hindi, Punjabi.
     Staff: English, Spanish,Hindi, Punjabi .
- Doctor: Dr. Meenakshi Tomar, DDS - 30 years experience, NYU School of Dentistry (2000), specializes in full mouth reconstruction, smile makeovers, laser surgery, WCLI certified
- Insurance: Accepts most PPO plans including UHC, Aetna, Premera, Delta Dental, MetLife, Blue Cross, Blue Shield, Anthem, Lifewise, Cigna, Humana, Ameritas, United Concordia, Careington, Spirit Dental. Does NOT accept Medicare or Apple Health.
- Payments: Cash, major credit cards, insurance. NO bitcoin.
- Policies: NO laughing gas, botox, silver fillings, or amalgam- she prefer composite feelings. Treats toddlers. No hygienist on staff. Bathroom for patients only.
- Cost/Price : its varies based on the procedure, complexity, and insurance coverage. Please call the clinic for specific pricing information.

For general dental questions, you may answer using standard dental knowledge.
For treatment-specific clinic questions, use only provided context.
Never answer non-dental questions.
Be very short, natural, and friendly.
Maximum 2 short sentences only.
Never give long explanations.
Respond only in {lang}.
"""
        )

        self.qdrant_client = qdrant_client
        self.openai_client = openai_client or OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.collection_name = "dental_knowledge"
        self.prefetched_embedding: asyncio.Task | None = None

    async def classify_query(self, query: str):
        q = query.lower()
        for word in MINIMAL_HARD_RULES:
            if word in q:
                return {"type": "non_dental"}

        prompt = f"""
You are a STRICT intent classifier for a dental clinic voice assistant.

Your job is to classify the user query into EXACTLY ONE of these categories:

1. clinic_hours
→ questions about clinic opening, closing, timings, schedule

2. appointment
→ booking, scheduling, or requesting a visit

3. rag
→ ANY dental or clinic-related question
This includes dental problems, visit another state/out of the state query,contact no,doctor tomar profile.location,bitcoin, treatments, doctor info, clinic info, insurance, payment, Bitcoin, cost/price ,hygienist, general dental knowledge.
If the query is EVEN SLIGHTLY related to dentistry, oral health, clinic services, or doctors, classify as "rag".

4. non_dental
→ ONLY if completely unrelated to dentistry or dental clinics
Examples: prime minister, weather, cricket, coding,computer,books, news, sports, movies, music, celebrities etc.

STRICT RULES:
- Always return EXACTLY ONE category
- Never explain anything
- Never answer the user
- Output ONLY valid JSON

Output format:
{{"type":"rag"}}

User Query:
{query}
"""

        GROQ_TO_ROUTE = {
            "clinic_hours": "clinic_hours",
            "appointment": "appointment",
            "rag": "general_dental",
            "non_dental": "non_dental",
        }

        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: self.groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=20,
                    )
                ),
                timeout=1.5,
            )
            raw = res.choices[0].message.content.strip()
            try:
                parsed = json.loads(raw)
                route = GROQ_TO_ROUTE.get(parsed.get("type", ""), "general_dental")
                return {"type": route}
            except:
                return {"type": "general_dental"}
        except:
            return {"type": "general_dental"}

    def build_context(self, query_type: str, tool_result, rag_result) -> str:
        parts = [f"query_type: {query_type}"]

        if tool_result:
            parts.append(f"tool_result: {tool_result}")

        if rag_result and rag_result.get("status") == "ok":
            parts.append(f"rag_result: {rag_result['context']}")

        lang_instructions = {
            "hi": "Hindi",
            "pa": "Punjabi",
            "es": "Spanish",
            "de": "German",
            "fr": "French",
        }
        if self._lang in lang_instructions:
            lang_name = lang_instructions[self._lang]
            parts.append(
                f"IMPORTANT: Internal context may be in English. "
                f"But final response MUST be strictly in {lang_name} only.include day, time if relevant. Never reply in English."
                f"Never reply in English."
            )
        else:
            parts.append("IMPORTANT: Reply strictly in English only.")

        return "\n".join(parts)

    async def on_user_turn_completed(self, turn_ctx, new_message):
        turn_start = time.perf_counter()
        query = ""
        if hasattr(new_message, "content") and new_message.content:
            first = new_message.content[0]
            if isinstance(first, str):
                query = first.strip()
            else:
                query = str(first).strip()
        logger.info(f"TRANSCRIPT RAW: {query}")

        # start embedding immediately, reuse prefetch if available
        if self.prefetched_embedding and not self.prefetched_embedding.done():
            embed_task = self.prefetched_embedding
        else:
            embed_task = asyncio.create_task(
                asyncio.to_thread(lambda: self.openai_client.embeddings.create(
                    input=query, model="text-embedding-3-small"
                ).data[0].embedding)
            )
        self.prefetched_embedding = None

        # ROUTING with latency log
        t0 = time.perf_counter()
        route_type = fast_classify(query)
        route_task = None
        if not route_type:
            route_task = asyncio.create_task(self.classify_query(query))
        if route_task:
            route_type = (await route_task)["type"]
        logger.info(f"ROUTING LATENCY: {time.perf_counter()-t0:.3f}s | ROUTE: {route_type}")

        tool_result = None
        rag_result = None

        if route_type == "non_dental":
            tool_result = (
                "I'm the dental clinic assistant and can only help with "
                "dental-related questions, appointments, clinic information, "
                "and treatment guidance."
            )
        elif route_type == "clinic_hours":
            tool_result = await self.check_clinic_hours()
        elif route_type == "appointment":
            clinic_status = await self.check_clinic_hours()
            tool_result = f"""
{clinic_status}

I can't book appointments directly.

Please call our scheduling team at 425-775-5162 to schedule your visit.
"""
        else:
            async def rag_pipeline():
                t_emb = time.perf_counter()
                emb = await embed_task
                logger.info(f"EMBED LATENCY: {time.perf_counter()-t_emb:.3f}s")
                return await self.knowledge_search(query, route_type, emb=emb)

            rag_task = asyncio.create_task(rag_pipeline())
            rag_result = await rag_task

        # Inject context then hand off to LiveKit — llm_node handles streaming
        ctx_str = self.build_context(route_type, tool_result, rag_result)
        logger.info(f"context_len: {len(ctx_str)}")
        turn_ctx.add_message(role="system", content=ctx_str)
        t_llm = time.perf_counter()
        await super().on_user_turn_completed(turn_ctx, new_message)
        logger.info(f"LLM HANDOFF after {time.perf_counter()-t_llm:.3f}s")
        logger.info(f"TOTAL TURN LATENCY: {time.perf_counter()-turn_start:.3f}s")

    async def llm_node(self, chat_ctx, tools, model_settings):
        t_first = time.perf_counter()
        first_token = True
        full_response = ""
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            try:
                token = ""
                if hasattr(chunk, "delta") and chunk.delta:
                    if hasattr(chunk.delta, "content"):
                        token = chunk.delta.content or ""
                if token:
                    full_response += token
            except Exception as e:
                logger.warning(f"TOKEN ERROR: {e}")
            if first_token:
                logger.info(f"LLM FIRST TOKEN: {time.perf_counter()-t_first:.3f}s")
                first_token = False
            yield chunk
        logger.info(f"LLM FINAL RESPONSE: {full_response}")

    async def tts_node(self, text, model_settings):
        if self._lang in ["hi", "pa"] and isinstance(text, str):
            text = clean_text(text)
        tts_start = time.perf_counter()
        logger.info("TTS REQUEST START")
        first_frame = True
        try:
            async for frame in Agent.default.tts_node(self, text, model_settings):
                if first_frame:
                    logger.info(f"TTS FIRST AUDIO FRAME: {time.perf_counter()-tts_start:.3f}s")
                    logger.info("AUDIO PLAYBACK START")
                    first_frame = False
                yield frame
        except Exception as e:
            logger.warning(f"TTS failed ({self._lang}), falling back to Deepgram: {e}")
            self._tts = deepgram.TTS(model="aura-2-thalia-en")
            first_frame = True
            async for frame in Agent.default.tts_node(self, text, model_settings):
                if first_frame:
                    logger.info(f"TTS FIRST AUDIO FRAME (fallback): {time.perf_counter()-tts_start:.3f}s")
                    logger.info("AUDIO PLAYBACK START (fallback)")
                    first_frame = False
                yield frame

    async def check_clinic_hours(self):
        """Check if the clinic is currently open based on Los Angeles time"""
        la_time = datetime.now(ZoneInfo("America/Los_Angeles"))
        current_day = la_time.strftime("%A")
        current_hour = la_time.hour
        logger.info(f"LA_TIME={la_time.isoformat()} | DAY={current_day} | HOUR={current_hour}")

        open_days = ["Monday", "Tuesday", "Thursday"]

        if current_day in open_days:
            if 7 <= current_hour < 18:
                return "Yes, the clinic is open now. We're open until 6 PM today."
            if current_hour < 7:
                return "The clinic is currently closed. We open at 7 AM today."
            next_open = {"Monday": "Tuesday", "Tuesday": "Thursday", "Thursday": "Monday"}
            return f"The clinic is closed now. We'll open on {next_open[current_day]} at 7 AM."

        next_open = {
            "Wednesday": "Thursday",
            "Friday": "Monday",
            "Saturday": "Monday",
            "Sunday": "Monday",
        }
        return f"The clinic is closed today. We'll be open on {next_open[current_day]} at 7 AM."

    async def knowledge_search(self, query: str, route_type: str, emb=None):
        """Search dental knowledge base"""
        if not self.qdrant_client:
            return {"status": "no_match"}

        async def _search(emb, q: str):
            return await asyncio.to_thread(
                lambda: self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=emb,
                    limit=5,
                    with_payload=True
                ).points
            )

        try:
            if emb is None:
                emb = await asyncio.to_thread(
                    lambda: self.openai_client.embeddings.create(
                        input=query, model="text-embedding-3-small"
                    ).data[0].embedding
                )

            t0 = time.perf_counter()
            results = await _search(emb, query)
            logger.info(f"RAG SEARCH LATENCY: {time.perf_counter()-t0:.3f}s")

            score = results[0].score if results else 0

            if not results or score < 0.40:
                return {"status": "no_match"}

            context = " ".join([r.payload["content"] for r in results])[:300]
            return {"status": "ok", "context": context}

        except Exception as e:
            logger.error(f"Qdrant error: {e}")
            return {"status": "no_match"}


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["tts_en"] = cartesia.TTS(
        model="sonic-2",
        voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        language="en",
    )
    proc.userdata["openai"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    proc.userdata["groq"] = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        proc.userdata["qdrant"] = QdrantClient(path="./qdrant_db", force_disable_check_same_thread=True)
    except Exception as e:
        logger.warning(f"Prewarm Qdrant failed: {e}")
        proc.userdata["qdrant"] = None


server.setup_fnc = prewarm


@server.rtc_session()
async def voice_agent(ctx: JobContext):
    logger.info(f"AGENT TRIGGERED - Agent joining room: {ctx.room.name}")

    try:
        await ctx.connect(auto_subscribe="audio_only")
        logger.info(f"Agent connected to room {ctx.room.name}")

        metadata = json.loads(ctx.room.metadata or "{}")
        raw_lang = metadata.get("lang", "en")
        lang = normalize_lang(raw_lang)
        logger.info(f"""
RAW LANG: {raw_lang}
NORMALIZED LANG: {lang}
STT: {"SARVAM" if lang in ["hi", "pa"] else "DEEPGRAM"}
TTS: {"SARVAM" if lang in ["hi", "pa"] else "DEEPGRAM"}
""")

        @ctx.room.on("participant_connected")
        def on_participant_connected(participant):
            logger.info(f"Participant connected: {participant.identity}")

        @ctx.room.on("track_subscribed")
        def debug_audio(track, *_):
            logger.info("RAW AUDIO RECEIVED")

        t_session = time.perf_counter()
        if lang in ["hi", "pa"]:
            llm_model = openai.LLM(
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            llm_model = openai.LLM(
                model="qwen/qwen3-32b",
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )

        session = AgentSession(
            stt=select_stt(lang),
            llm=llm_model,
            tts=ctx.proc.userdata["tts_en"] if lang == "en" else select_tts(lang),
            vad=ctx.proc.userdata["vad"],
            preemptive_generation=True,
        )
        logger.info(f"Session init latency: {time.perf_counter()-t_session:.3f}s")

        @session.on("user_speech_committed")
        def on_user_speech_committed(msg):
            logger.info("USER SPEECH COMMITTED — processing turn")

        @session.on("agent_speech_interrupted")
        def on_agent_speech_interrupted():
            logger.info("AGENT INTERRUPTED by user")

        @session.on("agent_speech_started")
        def on_agent_speech_started():
            logger.info("LLM FIRST TOKEN → AGENT SPEECH STARTED (TTS pipeline triggered)")

        @session.on("user_input_transcribed")
        def on_partial_transcript(ev):
            agent = session._agent
            if not isinstance(agent, Assistant):
                return
            text = ev.transcript.strip()
            if not text:
                return
            if not ev.is_final:
                if agent.prefetched_embedding is None or agent.prefetched_embedding.done():
                    agent.prefetched_embedding = asyncio.create_task(
                        asyncio.to_thread(lambda t=text: agent.openai_client.embeddings.create(
                            input=t, model="text-embedding-3-small"
                        ).data[0].embedding)
                    )
                    logger.info(f"PARTIAL STT prefetch: {text[:40]}")
            elif len(text) > 20:
                logger.info(f"EARLY TRIGGER on final: {text[:40]}")

        await session.start(
            agent=Assistant(
                lang=lang,
                qdrant_client=ctx.proc.userdata.get("qdrant"),
                openai_client=ctx.proc.userdata.get("openai"),
            ),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    sample_rate=16000,
                    num_channels=1,
                    noise_cancellation=lambda _: noise_cancellation.BVC()
                )
            ),
        )

        logger.info("Agent session started successfully")
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
