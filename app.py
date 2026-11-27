from __future__ import annotations

import streamlit as st

import config
from main import AnalysisPipeline
from modules.audio_input import AudioValidationError, load_audio_bytes
from modules.sentiment_analysis import SentimentError
from modules.speech_to_text import TranscriptionError
from modules.visualization import LABELS_LT, record_to_csv, record_to_json, save_results

st.set_page_config(page_title="Kalba ir sentimentas", layout="centered")


@st.cache_resource(show_spinner=False, max_entries=1)
def get_pipeline(whisper_model: str, sentiment_model: str) -> AnalysisPipeline:
    pipeline = AnalysisPipeline(whisper_model, sentiment_model)
    pipeline.start_background_load()
    return pipeline


@st.fragment(run_every=2)
def model_status(pipeline: AnalysisPipeline, downloading: bool) -> None:
    if pipeline.ready:
        st.caption("Modeliai paruošti")
    elif downloading:
        st.info(" Modeliai parsisiunčiami ir įkeliami fone (tik pirmą kartą, kelios minutės). "
                "Garsą galite įkelti ar įrašyti jau dabar.")
    else:
        st.info("Modeliai įkeliami fone. Garsą galite įkelti ar įrašyti jau dabar.")


def current_audio_source():
    mode = st.radio("Garso šaltinis", ["Failas", "Mikrofonas"], horizontal=True)
    if mode == "Failas":
        uploaded = st.file_uploader(
            "Įkelkite garso failą",
            type=[ext.lstrip(".") for ext in config.ALLOWED_EXTENSIONS],
        )
        if uploaded is not None:
            st.audio(uploaded)
            return uploaded.getvalue(), uploaded.name
    else:
        recording = st.audio_input("Įrašykite kalbą")
        if recording is not None:
            return recording.getvalue(), "microphone.wav"
    return None, None


st.title("🎙️ Kalbos atpažinimas ir sentimentų analizė")

with st.sidebar:
    st.header("Nustatymai")
    language_label = st.selectbox("Kalba", list(config.LANGUAGE_CHOICES))
    whisper_model = st.selectbox(
        "Whisper modelis",
        config.WHISPER_MODEL_CHOICES,
        index=config.WHISPER_MODEL_CHOICES.index(config.WHISPER_MODEL_NAME)
        if config.WHISPER_MODEL_NAME in config.WHISPER_MODEL_CHOICES else 2,
    )
    sentiment_key = st.selectbox(
        "Sentimentų modelis",
        list(config.SENTIMENT_MODELS),
        help="„multilingual“ tinka lietuvių ir anglų kalboms, „english“ tik anglų.",
    )
    st.caption(f"Įrenginys: {config.get_device()}")

language = config.LANGUAGE_CHOICES[language_label]
_models_on_disk = config.whisper_model_cached(whisper_model) and config.sentiment_model_cached(
    config.SENTIMENT_MODELS[sentiment_key]
)
pipeline = get_pipeline(whisper_model, sentiment_key)
model_status(pipeline, downloading=not _models_on_disk)

audio_bytes, audio_name = current_audio_source()

if st.button("1. Transkribuoti", disabled=audio_bytes is None, type="primary"):
    try:
        with st.spinner("Atpažįstama kalba..." if pipeline.stt.ready
                        else "Laukiama, kol įsikels modelis, tada atpažįstama kalba..."):
            audio = load_audio_bytes(audio_bytes, audio_name)
            transcription = pipeline.transcribe(audio, language)
        st.session_state["audio"] = audio
        st.session_state["transcription"] = transcription
        st.session_state["transcript_text"] = transcription.text
        st.session_state.pop("sentiment", None)
    except (AudioValidationError, TranscriptionError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Netikėta klaida: {exc}")

transcription = st.session_state.get("transcription")
if transcription:
    st.subheader("Atpažintas tekstas")
    st.caption(
        f"Nustatyta kalba: {transcription.language} · modelis: {transcription.model_name} "
        f"· trukmė: {transcription.duration:.1f} s"
    )
    if transcription.language not in config.SUPPORTED_LANGUAGE_CODES:
        st.warning("Nustatyta kalba nėra lietuvių ar anglų, sentimento rezultatas gali būti netikslus.")
    st.text_area("Tekstą galima pataisyti prieš analizę", key="transcript_text", height=140)

    if st.button("2. Analizuoti sentimentą"):
        try:
            with st.spinner("Analizuojamas sentimentas..." if pipeline.sentiment.ready
                            else "Laukiama, kol įsikels modelis, tada analizuojama..."):
                result = pipeline.analyze_text(st.session_state["transcript_text"])
            audio = st.session_state["audio"]
            record = pipeline.make_record(
                source=audio.source_name, duration=audio.duration,
                language=transcription.language,
                transcript=st.session_state["transcript_text"].strip(), result=result,
            )
            paths = save_results(record)
            st.session_state["sentiment"] = (result, record, paths)
        except SentimentError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Netikėta klaida: {exc}")

saved = st.session_state.get("sentiment")
if saved:
    result, record, paths = saved
    st.subheader("Rezultatai")
    st.metric("Dominuojantis sentimentas", LABELS_LT[result.label], f"{result.confidence:.1f}%", delta_color="off")
    st.table({
        "Klasė": [LABELS_LT[k] for k in result.scores],
        "Tikimybė, %": [f"{v:.2f}" for v in result.scores.values()],
    })
    left, right = st.columns(2)
    left.download_button("Atsisiųsti CSV", record_to_csv(record), "analysis.csv", "text/csv")
    right.download_button("Atsisiųsti JSON", record_to_json(record), "analysis.json", "application/json")
    st.caption(f"Rezultatai išsaugoti: {paths['json'].parent}")