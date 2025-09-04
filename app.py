# app.py
import os
import io
import time
import uuid
import base64
import tempfile
from typing import Optional, Dict, List

import numpy as np
import streamlit as st

# Load the CSS
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("EchoVerse — AI Audiobook")
st.write("Paste or upload your text below...")


# Audio + pipelines
from transformers import pipeline
import soundfile as sf
from pydub import AudioSegment

# Fallback TTS (optional, smaller dependency)
try:
    from gtts import gTTS  # noqa: F401
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

st.set_page_config(page_title="EchoVerse — AI Audiobook", layout="wide")

# -------------------------- Styling --------------------------
def load_css():
    css_path = os.path.join("static", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def pastel_bg():
    bg_svg = """
<svg width="1600" height="900" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#e3f2fd"/>
      <stop offset="50%" stop-color="#fce4ec"/>
      <stop offset="100%" stop-color="#e8f5e9"/>
    </linearGradient>
  </defs>
  <rect width="1600" height="900" fill="url(#g)"/>
  <circle cx="200" cy="150" r="120" fill="#fff" fill-opacity="0.25"/>
  <circle cx="1400" cy="700" r="200" fill="#fff" fill-opacity="0.18"/>
</svg>
""".strip()
    b64 = base64.b64encode(bg_svg.encode()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/svg+xml;base64,{b64}");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

load_css()
pastel_bg()

st.title("EchoVerse — AI Audiobook")

with st.expander("About & security", expanded=False):
    st.write(
        "Open-source prototype using Hugging Face models. "
        "Set HF_API_TOKEN in your environment for more reliable model pulls. "
        "Never paste private API keys into the UI."
    )

# -------------------------- Globals --------------------------
HF_TOKEN = os.getenv("HF_API_TOKEN", None)
MAX_UPLOAD_MB = 200
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

VOICE_PRESETS = {
    "Lisa": {"speaker_id": "female-en-1"},
    "Michael": {"speaker_id": "male-en-1"},
    "Allison": {"speaker_id": "female-en-2"},
    "Kate": {"speaker_id": "female-en-3"},
}

TONE_GUIDANCE = {
    "Neutral": "Rewrite neutrally and clearly, preserving all facts.",
    "Suspenseful": "Increase tension and anticipation with vivid pacing, keep semantics.",
    "Inspiring": "Make it uplifting and motivational while preserving meaning.",
    "Humorous": "Lightly humorous and playful without changing facts.",
    "Formal": "Make it formal, precise, and professional without altering meaning.",
}

LANG_TO_CODE = {"English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta"}

# -------------------------- Pipelines (lazy) --------------------------
@st.cache_resource(show_spinner=False)
def get_asr():
    try:
        return pipeline(
            "automatic-speech-recognition",
            model="ibm-granite/granite-speech-3.3-2b",
            use_auth_token=HF_TOKEN,
        )
    except Exception as e:
        st.warning(f"ASR init failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_rewriter():
    # Prefer Granite-8B-Instruct; fallback to FLAN-T5 base (lighter)
    try:
        return pipeline(
            "text-generation",
            model="ibm-granite/granite-8b-instruct",
            use_auth_token=HF_TOKEN,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
        )
    except Exception as e:
        st.warning(f"Granite rewrite unavailable ({e}); falling back to FLAN-T5.")
        try:
            return pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_new_tokens=256,
            )
        except Exception as e2:
            st.error(f"Rewrite fallback failed: {e2}")
            return None

@st.cache_resource(show_spinner=False)
def get_tts():
    # Try Coqui XTTS-v2 via transformers; if it fails, we will fallback to gTTS
    try:
     return pipeline("text-to-speech", model="tts_models/en/ljspeech/tacotron2-DDC")
    except Exception as e:
        st.warning(f"TTS (XTTS-v2) unavailable: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_translator(direction: str):
    model_map = {
        "en->hi": "Helsinki-NLP/opus-mt-en-hi",
        "hi->en": "Helsinki-NLP/opus-mt-hi-en",
        "en->ta": "Helsinki-NLP/opus-mt-en-ta",
        "ta->en": "Helsinki-NLP/opus-mt-ta-en",
        "en->te": "Helsinki-NLP/opus-mt-en-te",
        "te->en": "Helsinki-NLP/opus-mt-te-en",
    }
    model = model_map.get(direction)
    if not model:
        return None
    try:
        return pipeline("translation", model=model, use_auth_token=HF_TOKEN)
    except Exception as e:
        st.warning(f"Translator init failed ({direction}): {e}")
        return None

# -------------------------- Helpers --------------------------
def ensure_size_limit(b: bytes) -> bool:
    return len(b) <= MAX_UPLOAD_BYTES

def rewrite_text(original: str, tone: str) -> str:
    p = get_rewriter()
    if not p:
        return "(Rewrite model unavailable)"
    guidance = TONE_GUIDANCE.get(tone, TONE_GUIDANCE["Neutral"])

    # Different prompt depending on pipeline type
    try:
        # text-generation style (Granite)
        if getattr(p, "task", "") == "text-generation" or "text-generation" in str(p):
            prompt = (
                f"You are a skilled editor. {guidance}\n\n"
                f"Original:\n{original}\n\nRewritten:"
            )
            out = p(prompt)[0].get("generated_text", "")
            if "Rewritten:" in out:
                out = out.split("Rewritten:", 1)[-1].strip()
            return out.strip()
        else:
            # text2text-generation (FLAN)
            prompt = f"{guidance}\nRewrite the following text without changing the meaning:\n{original}"
            out = p(prompt)[0].get("generated_text", "")
            return out.strip()
    except Exception as e:
        st.error(f"Rewrite failed: {e}")
        return ""

def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "English":
        return text
    dir_map = {"Hindi": "en->hi", "Telugu": "en->te", "Tamil": "en->ta"}
    translator = get_translator(dir_map[target_lang])
    if not translator:
        return text
    try:
        return translator(text)[0].get("translation_text", text)
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

def audio_from_wav_array(wav: np.ndarray, sr: int = 24000, bitrate: str = "192k") -> bytes:
    # Convert a float32 wav array to MP3 bytes via pydub/ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        sf.write(wf.name, wav, sr)
        audio = AudioSegment.from_wav(wf.name)
        buf = io.BytesIO()
        audio.export(buf, format="mp3", bitrate=bitrate)
        buf.seek(0)
        return buf.read()

def tts(text: str, language: str, voice_name: str) -> Optional[bytes]:
    # 1) Try transformers TTS (XTTS-v2). 2) Fallback to gTTS if available.
    lang_code = LANG_TO_CODE.get(language, "en")

    # Try XTTS-v2 first
    xtts = get_tts()
    if xtts is not None:
        try:
            params = VOICE_PRESETS.get(voice_name, {})
            out = xtts(text, language=lang_code, **params)
            if isinstance(out, dict) and "audio" in out:
                wav = np.array(out["audio"], dtype=np.float32)
                sr = int(out.get("sampling_rate", 24000))
            elif isinstance(out, list) and len(out) > 0 and "audio" in out[0]:
                wav = np.array(out[0]["audio"], dtype=np.float32)
                sr = int(out[0].get("sampling_rate", 24000))
            else:
                raise ValueError("Unexpected TTS output format")
            return audio_from_wav_array(wav, sr)
        except Exception as e:
            st.warning(f"XTTS synthesis failed: {e}")

    # Fallback: gTTS (requires internet, but no API key; not Google Cloud)
    if HAS_GTTS:
        try:
            tts_obj = gTTS(text=text, lang=lang_code if lang_code in ["en", "hi", "ta", "te"] else "en")
            buf = io.BytesIO()
            tts_obj.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            st.error(f"gTTS fallback failed: {e}")
            return None
    else:
        st.error("No TTS backend available. Install gTTS or try setting HF_API_TOKEN for XTTS-v2.")
        return None

def add_history(rec: Dict):
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].insert(0, rec)

# -------------------------- Sidebar Controls --------------------------
with st.sidebar:
    st.header("Controls")
    tone = st.selectbox("Narrative tone", list(TONE_GUIDANCE.keys()), index=0)
    voice = st.selectbox("Voice", list(VOICE_PRESETS.keys()), index=0)
    out_lang = st.selectbox("Output language", ["English", "Hindi", "Telugu", "Tamil"], index=0)
    st.markdown("---")
    st.subheader("Chat with EchoVerse")
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    for role, msg in st.session_state["chat"]:
        st.markdown(f"**{'You' if role=='user' else 'EchoVerse'}:** {msg}")
    chat_in = st.text_input("Ask or refine the narration…", key="chat_input")
    if st.button("Send", key="send_btn"):
        if chat_in.strip():
            st.session_state["chat"].append(("user", chat_in))
            helper = get_rewriter()
            if helper:
                try:
                    if getattr(helper, "task", "") == "text-generation" or "text-generation" in str(helper):
                        resp = helper(f"Be concise and helpful. Q: {chat_in}\nA:")[0].get("generated_text", "")
                    else:
                        resp = helper(f"Answer concisely:\n{chat_in}")[0].get("generated_text", "")
                except Exception:
                    resp = "(LLM unavailable right now)"
            else:
                resp = "(LLM unavailable right now)"
            st.session_state["chat"].append(("assistant", resp))
            st.experimental_rerun()

# -------------------------- Component 1: Input --------------------------
st.markdown('<div class="card"><div class="section-title">Component 1: Text Input & File Upload</div>', unsafe_allow_html=True)
col1, col2 = st.columns([2,1])

with col1:
    user_text = st.text_area("Paste text here", height=180, placeholder="Paste or type your text…")
    up = st.file_uploader("Or drop a .txt file (≤ 200 MB)", type=["txt"], accept_multiple_files=False)
    if up:
        data = up.getvalue()
        if ensure_size_limit(data):
            try:
                user_text = data.decode("utf-8", errors="ignore")
                st.success(f"Parsed file '{up.name}' ({len(data)/1024:.1f} KB)")
            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            st.error("File exceeds 200 MB limit.")

with col2:
    st.write("Optional: Upload audio for Speech→Text (ASR)")
    up_aud = st.file_uploader("Audio (wav/mp3/m4a/ogg)", type=["wav","mp3","m4a","ogg"], accept_multiple_files=False)
    if up_aud:
        with st.spinner("Transcribing with IBM Granite ASR…"):
            asr = get_asr()
            if asr:
                try:
                    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
                    tmp.write(up_aud.getvalue()); tmp.flush()
                    res = asr(tmp.name)
                    text = res.get("text") if isinstance(res, dict) else str(res)
                    user_text = (user_text + "\n" + text) if user_text else text
                    st.success("Transcription complete and appended to input.")
                except Exception as e:
                    st.error(f"ASR failed: {e}")
            else:
                st.error("ASR pipeline unavailable.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------- Components 2 & 3 --------------------------
st.markdown('<div class="card"><div class="section-title">Components 2 & 3: Tone/Voice & Original vs Rewritten</div>', unsafe_allow_html=True)
do_generate = st.button("Generate Audiobook (rewrite → translate → synthesize)")
rewritten_text = st.session_state.get("rewritten_text", "")

if do_generate and not user_text:
    st.error("Please provide input text or a .txt file.")

if do_generate and user_text:
    with st.spinner("Rewriting with LLM…"):
        rewritten_text = rewrite_text(user_text, tone)
        st.session_state["rewritten_text"] = rewritten_text

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Original**")
    st.markdown('<div class="colbox">', unsafe_allow_html=True)
    st.write(user_text if user_text else "(No input yet)")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown("**Rewritten**")
    st.markdown('<div class="colbox">', unsafe_allow_html=True)
    st.write(rewritten_text if rewritten_text else "(Click Generate to rewrite)")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------- Component 4: Audio --------------------------
st.markdown('<div class="card"><div class="section-title">Component 4: Audio Playback & Download</div>', unsafe_allow_html=True)
if st.button("Synthesize narration to MP3"):
    text_for_tts = st.session_state.get("rewritten_text") or user_text
    if not text_for_tts:
        st.error("Nothing to synthesize yet. Provide text and click Generate first.")
    else:
        with st.spinner("Synthesizing speech…"):
            # Translate before TTS if needed
            # (Assume source is English after rewrite; could add auto-detect, omitted for simplicity)
            translated = translate_text(text_for_tts, out_lang)
            mp3 = tts(translated, out_lang, voice)
            if mp3:
                nid = str(uuid.uuid4())
                rec = {
                    "id": nid,
                    "tone": tone,
                    "voice": voice,
                    "language": out_lang,
                    "rewritten_text": text_for_tts,
                    "audio_mp3": mp3,
                }
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].insert(0, rec)
                st.audio(mp3, format="audio/mp3")
                st.download_button("Download MP3", data=mp3, file_name=f"echoverse_{nid}.mp3", mime="audio/mpeg")
            else:
                st.error("TTS failed; see messages above.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------- Component 5: History --------------------------
st.markdown('<div class="card"><div class="section-title">Component 5: Past Narrations</div>', unsafe_allow_html=True)
hist = st.session_state.get("history", [])
if not hist:
    st.info("No narrations yet. Generate one above.")
else:
    with st.expander("Show past narrations", expanded=True):
        for rec in hist:
            st.markdown(
                f"<div class='past-item'><b>Tone:</b> {rec['tone']} &nbsp; | &nbsp; "
                f"<b>Voice:</b> {rec['voice']} &nbsp; | &nbsp; <b>Language:</b> {rec['language']}</div>",
                unsafe_allow_html=True,
            )
            st.write(rec["rewritten_text"])
            st.audio(rec["audio_mp3"], format="audio/mp3")
            st.download_button("Re-download MP3", data=rec["audio_mp3"], file_name=f"echoverse_{rec['id']}.mp3", mime="audio/mpeg")
            st.markdown("---")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------- Notes --------------------------
with st.expander("Notes on responsiveness & testing"):
    st.write(
        "On CPUs, large models may exceed 6–8s. If models fail to load, the app falls back where possible "
        "(e.g., FLAN-T5 for rewrite, gTTS for synthesis). Install FFmpeg to enable MP3 export."
    )
