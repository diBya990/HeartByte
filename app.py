import os
import io
import json
import time
import base64
import numpy as np
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go

from data_loader import load_audio_file, get_available_audio_files
from signal_processor import (
    bandpass_filter, extract_envelope, analyze_peaks, generate_synthetic_ecg,
    downsample_preserve_peaks, get_interval_stats, label_heart_sounds,
)
from condition_classifier import evaluate_heart_condition, EDUCATIONAL_DISCLAIMER
from report_generator import generate_pdf_report, generate_comparison_pdf_report
from heartbeat_classifier import predict_is_heartbeat

# Page Setup
st.set_page_config(page_title="Heartbeat Signal Viewer", page_icon="🫀", layout="wide")

# Streamlit's file_uploader shows an "Add files" (+) button next to an already-uploaded
# file even in single-file mode; hide it since only one custom recording is ever kept
# (the "x" remove button still works to swap files). Also defines the pulsing-heart
# keyframes once, shared by every pulsing_heart_html() instance on the page.
st.markdown(
    """
    <style>
    button[aria-label="Add files"] { display: none; }
    /* Same "Systole Magenta" gradient + theme on every page (not just the
       front/home screens): the .streamlit/config.toml dark-magenta theme
       colors every native widget (buttons, sliders, dataframes, sidebar...)
       automatically; this just layers the gradient on top, since gradients
       aren't expressible through Streamlit's theme config. */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a0b2e 0%, #7a0e6e 45%, #ff2d8e 100%) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] { background: rgba(26,11,46,0.45) !important; }
    /* Sized for the narrower case (sidebar visible, on the feature pages), which
       is why the vw factor looks conservative on the sidebar-free home page. */
    h1 { white-space: nowrap !important; font-size: clamp(0.9rem, 1.65vw, 1.85rem) !important; }
    @keyframes heartPulse {
        0%   { transform: scale(1); }
        15%  { transform: scale(1.28); }
        30%  { transform: scale(1); }
        45%  { transform: scale(1.15); }
        60%  { transform: scale(1); }
        100% { transform: scale(1); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes breathe {
        0%, 100% { opacity: 0.45; letter-spacing: 1px; }
        50% { opacity: 1; letter-spacing: 2.5px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def pulsing_heart_html(bpm, size="1.6rem"):
    """A heart emoji whose CSS pulse animation duration matches the recording's actual BPM."""
    safe_bpm = bpm if bpm and bpm > 0 else 60.0
    safe_bpm = max(30.0, min(220.0, safe_bpm))
    duration = round(60.0 / safe_bpm, 3)
    return (
        f'<span style="display:inline-block; font-size:{size}; transform-origin:center; '
        f'animation: heartPulse {duration}s ease-in-out infinite;">🫀</span>'
    )


def _inject_home_theme():
    """
    No-scroll, single-viewport layout for the front/home screens (idle, pulsing,
    and menu stages, all inside render_home()), plus a few softly floating glass
    shapes (and a couple of blood-drop "stickers") behind the content. The
    "Systole Magenta" gradient itself is set globally (top of the file) so
    every page shares the same background — this only adds what's specific to
    this single-viewport home screen.

    The actual scrolling element in this Streamlit version is
    [data-testid="stMain"] (it carries `overflow-y: auto` by default), not the
    legacy `.main`/`.block-container` classes — those are targeted too as a
    harmless no-op fallback, but stMain/stMainBlockContainer are what matter.
    """
    st.markdown(
        """
        <style>
        html, body { overflow: hidden !important; }
        [data-testid="stAppViewContainer"] {
            overflow: hidden !important; height: 100vh !important; position: relative !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stMain"] { overflow: hidden !important; height: 100vh !important; }
        [data-testid="stMainBlockContainer"], .main .block-container {
            overflow: hidden !important;
            height: 100vh !important;
            padding-top: 1.4rem !important;
            max-width: 100% !important;
        }
        /* Streamlit gives every element-container `position: relative` by
           default, which would otherwise become the (0-height) containing
           block for .hb-shapes' absolute inset:0 below. Neutralizing it here
           lets that search continue up to stAppViewContainer instead, which
           now correctly spans the full 100vh. */
        [data-testid="stElementContainer"] { position: static !important; }
        h1 { color: #f9ecf5 !important; text-shadow: 0 2px 22px rgba(255,45,142,0.55); }

        @keyframes hbFloatA {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-22px) rotate(8deg); }
        }
        @keyframes hbFloatB {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(18px) rotate(-9deg); }
        }
        /* No explicit z-index here on purpose: an unpositioned/auto-z element
           simply paints in DOM order relative to its siblings, so as long as
           this markdown call runs before the heading/orb/card below, it stays
           visually behind them without needing to fight a stacking context
           (position:fixed or an explicit z-index would each pull this into
           its own stacking context, which then paints either entirely below
           stAppViewContainer's own background or entirely above its content —
           never sandwiched between the two, since a plain box's background
           and its descendants can't be split across stacking levels). */
        .hb-shapes { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
        .hb-shape { position: absolute; }
        .hb-sphere {
            border-radius: 50%;
            background: radial-gradient(circle at 32% 26%, rgba(255,255,255,0.55) 0%,
                        rgba(255,214,236,0.5) 18%, rgba(214,20,130,0.45) 45%,
                        rgba(122,14,110,0.35) 75%, rgba(26,11,46,0.15) 100%);
            box-shadow: 0 20px 55px rgba(122,14,110,0.35), inset -10px -10px 26px rgba(26,11,46,0.35);
        }
        .hb-sphere-pink {
            background: radial-gradient(circle at 32% 26%, rgba(255,255,255,0.5) 0%,
                        rgba(255,180,225,0.45) 20%, rgba(255,45,142,0.35) 50%, rgba(122,14,110,0.28) 100%);
        }
        /* A tilted "torus" ring (matches the reference mockup's cyan ring): a
           filled circle with a radial-gradient mask carved out of its middle,
           then tipped on its side with a perspective rotate. */
        .hb-ring {
            border-radius: 50%;
            background: radial-gradient(circle at 34% 28%, #ffffff 0%, #7cf5ff 42%, #7cf5ff 100%);
            -webkit-mask: radial-gradient(circle, transparent 48%, #000 53%, #000 84%, transparent 89%);
                    mask: radial-gradient(circle, transparent 48%, #000 53%, #000 84%, transparent 89%);
            transform: perspective(240px) rotateX(58deg) rotateZ(-12deg);
            filter: drop-shadow(0 10px 16px rgba(0,0,0,0.4));
        }
        .hb-diamond {
            clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
            background: linear-gradient(135deg, rgba(255,180,225,0.4), rgba(214,20,130,0.28));
        }
        /* Blood-drop "stickers": a small rotated glass chip pinning the 🩸
           emoji, like a decal, floating alongside the spheres/ring. */
        .hb-sticker {
            display: flex; align-items: center; justify-content: center;
            width: 42px; height: 42px; font-size: 1.35rem;
            background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.22);
            border-radius: 11px; box-shadow: 0 12px 24px rgba(0,0,0,0.35);
        }
        .hb-credit {
            display: inline-block; padding: 5px 18px; border-radius: 999px;
            background: rgba(255,180,225,0.14); border: 1px solid rgba(255,140,210,0.3);
            color: #ffe4f2; font-size: 0.72rem; font-weight: 700;
            letter-spacing: 1.6px; text-transform: uppercase;
        }
        </style>
        <div class="hb-shapes">
            <div class="hb-shape hb-sphere" style="top:-6%; left:1%; width:110px; height:110px; animation: hbFloatA 9s ease-in-out infinite;"></div>
            <div class="hb-shape hb-sphere hb-sphere-pink" style="top:66%; left:5%; width:74px; height:74px; animation: hbFloatB 7s ease-in-out infinite;"></div>
            <div class="hb-shape hb-ring" style="top:-4%; right:0%; width:60px; height:60px; animation: hbFloatB 10s ease-in-out infinite;"></div>
            <div class="hb-shape hb-sphere" style="top:68%; right:12%; width:150px; height:150px; animation: hbFloatA 11s ease-in-out infinite;"></div>
            <div class="hb-shape hb-diamond" style="top:42%; left:47%; width:56px; height:56px; animation: hbFloatA 8s ease-in-out infinite;"></div>
            <div class="hb-shape hb-sphere hb-sphere-pink" style="top:30%; right:28%; width:40px; height:40px; animation: hbFloatB 6.5s ease-in-out infinite;"></div>
            <div class="hb-shape hb-sticker" style="top:22%; left:24%; transform: rotate(-14deg); animation: hbFloatA 8.5s ease-in-out infinite;">🩸</div>
            <div class="hb-shape hb-sticker" style="bottom:20%; right:26%; transform: rotate(11deg); animation: hbFloatB 9.5s ease-in-out infinite;">🩸</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_audio_bytes_and_mime(audio_player_source, filename):
    """audio_player_source is either raw bytes (custom uploads) or a file path (archive files)."""
    if isinstance(audio_player_source, (bytes, bytearray)):
        audio_bytes = bytes(audio_player_source)
    else:
        with open(audio_player_source, "rb") as f:
            audio_bytes = f.read()
    mime = "audio/mpeg" if os.path.splitext(filename)[1].lower() == ".mp3" else "audio/wav"
    return audio_bytes, mime


def render_synced_waveform(time_ds, values_ds, audio_bytes, mime, key):
    """
    Self-contained canvas + <audio> component: draws the (already downsampled)
    waveform once, then sweeps a playback cursor across it in sync with the
    embedded audio's real playback position.

    Streamlit's native st.audio and st.plotly_chart live in separate parts of
    the page with no shared playback clock, so there's no way to drive a
    Plotly cursor from the native player's position. This embeds its own
    lightweight <audio> element inside the same custom component as the
    canvas specifically so a single script can listen to that element's
    `timeupdate` events directly and redraw the cursor every frame.
    """
    b64_audio = base64.b64encode(audio_bytes).decode("ascii")
    duration = float(time_ds[-1]) if len(time_ds) else 1.0
    points_json = json.dumps(list(zip(np.asarray(time_ds).tolist(), np.asarray(values_ds).tolist())))

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ margin: 0; background: #0e1117; font-family: sans-serif; }}
        .wrap {{ padding: 6px 4px; }}
        canvas {{ width: 100%; height: 220px; display: block; background: #0e1117; border-radius: 6px; }}
        audio {{ width: 100%; margin-top: 10px; }}
    </style>
    </head>
    <body>
    <div class="wrap">
        <canvas id="wf_{key}"></canvas>
        <audio id="au_{key}" controls src="data:{mime};base64,{b64_audio}"></audio>
    </div>
    <script>
        const points = {points_json};
        const canvas = document.getElementById('wf_{key}');
        const audio = document.getElementById('au_{key}');
        const ctx = canvas.getContext('2d');
        const totalDuration = {duration};

        let minV = Infinity, maxV = -Infinity;
        for (const p of points) {{ if (p[1] < minV) minV = p[1]; if (p[1] > maxV) maxV = p[1]; }}
        if (minV === maxV) {{ minV -= 1; maxV += 1; }}
        const pad = (maxV - minV) * 0.1;
        minV -= pad; maxV += pad;

        function draw(currentTime) {{
            const w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            const progress = totalDuration > 0 ? Math.min(1, currentTime / totalDuration) : 0;
            const cursorX = progress * w;

            ctx.fillStyle = 'rgba(249, 115, 22, 0.10)';
            ctx.fillRect(0, 0, cursorX, h);

            ctx.beginPath();
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = Math.max(1, w / 900);
            points.forEach((p, i) => {{
                const x = (p[0] / totalDuration) * w;
                const y = h - ((p[1] - minV) / (maxV - minV)) * h;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }});
            ctx.stroke();

            ctx.beginPath();
            ctx.strokeStyle = '#f97316';
            ctx.lineWidth = Math.max(2, w / 500);
            ctx.moveTo(cursorX, 0);
            ctx.lineTo(cursorX, h);
            ctx.stroke();
        }}

        function resize() {{
            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            draw(audio.currentTime || 0);
        }}

        audio.addEventListener('timeupdate', () => draw(audio.currentTime));
        audio.addEventListener('seeking', () => draw(audio.currentTime));
        audio.addEventListener('play', () => draw(audio.currentTime));
        window.addEventListener('resize', resize);
        setTimeout(resize, 50);
    </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=290)


# Header & Mandatory Disclaimer
st.title("🫀 HeartByte: Interactive Heartbeat Signal Processing & ECG Simulation Studio")
#st.warning(f"⚠️ **Educational Disclaimer**: {EDUCATIONAL_DISCLAIMER}")

if "view" not in st.session_state:
    st.session_state.view = "home"
if "home_stage" not in st.session_state:
    st.session_state.home_stage = "idle"

STATUS_COLORS = {
    "Normal Indications": "#22c55e",
    "Acoustic Anomaly": "#eab308",
    "Abnormal Rate": "#ef4444",
    "Irregular Rhythm": "#ef4444",
}

RISK_ALERT = {
    "Normal Indications": st.success,
    "Acoustic Anomaly": st.warning,
    "Abnormal Rate": st.error,
    "Irregular Rhythm": st.error,
}

# Hover-help text for non-clinical visitors, reused across st.metric/st.slider
# `help=` params and as HTML `title=` attributes on custom-rendered elements.
GLOSSARY = {
    "bpm": "Beats Per Minute — how many heartbeats were detected per minute, averaged over the recording.",
    "hrv": "Heart Rate Variability — the spread (standard deviation) of the time gaps between consecutive "
           "heartbeats, in milliseconds (an SDNN-style metric). Higher values can indicate an irregular rhythm.",
    "beats": "The number of complete heart cycles (one S1+S2 pair each) detected in this recording.",
    "status": "A rule-based summary label from the Acoustic & Rhythm Checklist below — an educational "
               "indicator only, not a validated clinical diagnosis.",
    "lowcut": "Frequencies below this value are filtered out. Real heart sounds (S1/S2) mostly live above ~20 Hz.",
    "highcut": "Frequencies above this value are filtered out. Real heart sounds (S1/S2) mostly live below ~150-200 Hz.",
    "s1": "S1 (\"lub\") — the sound of the mitral/tricuspid valves closing at the start of systole.",
    "s2": "S2 (\"dub\") — the sound of the aortic/pulmonic valves closing at the end of systole.",
    "sample_rate": "How many audio samples per second the recording was analyzed at.",
    "duration": "Length of the recording actually analyzed.",
    "bandpass_filter": "The frequency range kept by the bandpass filter before analysis; everything outside "
                        "it is treated as noise.",
    "mean_interval": "The average time gap between consecutive detected heartbeats.",
    "interval_range": "The shortest and longest time gaps between consecutive heartbeats in this recording.",
    "dataset": "Where this recording came from — an archive sub-dataset folder, or a custom upload.",
}

# Keep only zoom in / zoom out on the chart toolbar (the fullscreen control is
# Streamlit's own chart chrome, not part of Plotly's modebar, so it's unaffected).
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "toImage", "zoom2d", "pan2d", "select2d", "lasso2d",
        "autoScale2d", "resetScale2d",
    ],
}

MAX_PLOT_POINTS = 2500


def _downsample_for_plot(values, time, max_points=MAX_PLOT_POINTS):
    """
    Thins a waveform down to at most max_points before handing it to Plotly.
    A 5s/22050Hz recording is 110k+ raw samples per trace — plotting that at
    full resolution (especially with several such charts on screen at once, as
    in compare mode) builds huge SVG path data and can exhaust browser memory.
    Analysis always runs on the full-resolution array; only the chart display
    is thinned, using the same peak-preserving decimation as the ECG monitor
    so transient spikes don't get smoothed away.
    """
    if len(values) <= max_points:
        return values, time
    factor = max(1, len(values) // max_points)
    usable_len = (len(values) // factor) * factor
    return downsample_preserve_peaks(values, factor), time[:usable_len:factor]


def select_audio_source(key_prefix, uploader_label="Or Upload Your Own Heartbeat Recording"):
    """
    Renders the sidebar controls for choosing one heartbeat recording — either a
    custom upload (validated by the trained classifier) or an archive file.
    Returns (dataset_label, filename, filepath_for_load, audio_player_source), or
    None if the user hasn't picked an archive recording yet (no default is
    auto-selected). Halts the app via st.stop() if the upload can't be read or
    isn't a real heartbeat.
    """
    custom_audio = st.sidebar.file_uploader(
        uploader_label,
        type=["wav", "mp3"],
        help="WAV or MP3. Analyzed with the exact same pipeline as the archive recordings below. "
             "The uploaded file is kept in memory only and is never written to the archive or disk.",
        key=f"uploader_{key_prefix}",
    )
    st.sidebar.caption(
        "⚠️ Only real heartbeat (stethoscope/PCG) recordings are supported. "
        "Music, speech, or other non-heartbeat audio will be rejected after an automatic check."
    )

    if custom_audio is not None:
        audio_bytes = custom_audio.getvalue()

        try:
            probe_signal, probe_sr = load_audio_file(io.BytesIO(audio_bytes))
        except Exception as e:
            st.error(f"Could not read '{custom_audio.name}' as audio: {e}")
            st.stop()

        try:
            is_heartbeat, diag = predict_is_heartbeat(probe_signal, probe_sr)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        if not is_heartbeat:
            st.error(
                f"⚠️ '{custom_audio.name}' does not appear to be a real heartbeat sound. "
                "Heart_Byte only supports stethoscope-style heartbeat (PCG) recordings — "
                "please upload a genuine heartbeat audio file."
            )
            st.stop()

        return "Custom Upload", custom_audio.name, io.BytesIO(audio_bytes), audio_bytes

    file_dict = get_available_audio_files("archive")

    selected_set = st.sidebar.selectbox(
        "Choose Sub-dataset Folder", list(file_dict.keys()), key=f"dataset_{key_prefix}",
    )
    available_files = file_dict[selected_set]

    if not available_files:
        st.error(f"No WAV files found in 'archive/{selected_set}'. Please check your folder structure.")
        st.stop()

    # Extract simple file names for cleaner dropdown
    file_map = {os.path.basename(f): f for f in available_files}
    selected_filename = st.sidebar.selectbox(
        "Select Recording File", list(file_map.keys()),
        index=None, placeholder="Select a recording…", key=f"file_{key_prefix}",
    )
    if selected_filename is None:
        return None

    selected_filepath = file_map[selected_filename]
    return selected_set, selected_filename, selected_filepath, selected_filepath


def run_analysis(filepath, filename, lowcut, highcut):
    """Runs the full signal-processing pipeline for one recording and returns a results dict."""
    try:
        signal, sr = load_audio_file(filepath)
    except Exception as e:
        st.error(f"Could not read '{filename}' as audio: {e}")
        st.stop()

    filtered_signal = bandpass_filter(signal, lowcut=lowcut, highcut=highcut, fs=sr)
    envelope = extract_envelope(filtered_signal, fs=sr)
    peaks, peak_times, bpm, hrv = analyze_peaks(envelope, fs=sr)
    s1_indices, s2_indices = label_heart_sounds(envelope, peaks, fs=sr)
    time = np.arange(len(signal)) / sr
    status, findings, checklist = evaluate_heart_condition(bpm, hrv, filtered_signal, fs=sr)
    interval_stats = get_interval_stats(peak_times)

    return {
        "signal": signal, "sr": sr, "filtered_signal": filtered_signal,
        "envelope": envelope, "peaks": peaks, "peak_times": peak_times, "time": time,
        "s1_indices": s1_indices, "s2_indices": s2_indices,
        "bpm": bpm, "hrv": hrv, "status": status, "findings": findings,
        "checklist": checklist, "interval_stats": interval_stats,
    }


def render_single_mode(selected_set, selected_filename, selected_filepath, audio_player_source, lowcut, highcut):
    result = run_analysis(selected_filepath, selected_filename, lowcut, highcut)
    signal, sr = result["signal"], result["sr"]
    filtered_signal, envelope = result["filtered_signal"], result["envelope"]
    peaks, peak_times, time = result["peaks"], result["peak_times"], result["time"]
    s1_indices, s2_indices = result["s1_indices"], result["s2_indices"]
    bpm, hrv, status = result["bpm"], result["hrv"], result["status"]
    findings, checklist, interval_stats = result["findings"], result["checklist"], result["interval_stats"]

    simulated_ecg = generate_synthetic_ecg(time, peak_times)

    st.markdown(
        f"{pulsing_heart_html(bpm)} &nbsp; **Live Heart Rate — {bpm:.1f} BPM**",
        unsafe_allow_html=True,
    )

    # Top Metrics Bar
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1.3])
    col1.metric("Estimated Heart Rate", f"{bpm:.1f} BPM", help=GLOSSARY["bpm"])
    col2.metric("Heart Rate Variability", f"{hrv:.1f} ms", help=GLOSSARY["hrv"])
    col3.metric("Detected Beats", f"{len(peaks)}", help=GLOSSARY["beats"])
    col4.markdown(
        f"""
        <div style="line-height: 1.3;" title="{GLOSSARY['status']}">
            <div style="font-size: 0.875rem; color: var(--text-color); opacity: 0.6;">Diagnostic Status</div>
            <div style="font-size: 1.4rem; font-weight: 600; color: {STATUS_COLORS.get(status, '#eab308')};
                        white-space: normal; word-break: break-word; margin-top: 2px;">
                {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Audio Player
    st.audio(audio_player_source)

    # Interactive Plots
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Time-Domain Audio Signal", "🌊 Envelope & Beat Peaks",
        "⚡ Live 60FPS ECG Monitor", "🎯 Synced Playback",
    ])

    signal_ds, time_ds_raw = _downsample_for_plot(signal, time)
    filtered_ds, time_ds_filt = _downsample_for_plot(filtered_signal, time)
    envelope_ds, time_ds_env = _downsample_for_plot(envelope, time)

    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=time_ds_raw, y=signal_ds, mode='lines', name='Raw Audio (PCG)', opacity=0.4, line=dict(color='gray')))
        fig1.add_trace(go.Scatter(x=time_ds_filt, y=filtered_ds, mode='lines', name='Filtered Audio', line=dict(color='orange'), yaxis='y2'))
        fig1.update_layout(
            title="Raw vs. Filtered Acoustic Signal",
            xaxis_title="Time (seconds)",
            yaxis=dict(title="Raw Amplitude"),
            yaxis2=dict(title="Filtered Amplitude", overlaying='y', side='right', showgrid=False),
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("Filtered trace uses its own auto-scaled axis (right) so cutoff changes stay visible even when the passband amplitude is much smaller than the raw signal.")

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=time_ds_env, y=envelope_ds, mode='lines', name='Smoothed Envelope', line=dict(color='cyan', width=2)))
        fig2.add_trace(go.Scatter(
            x=s1_indices / sr, y=envelope[s1_indices], mode='markers', name='S1 (lub)',
            marker=dict(size=11, color='#f97316', symbol='triangle-up'),
            hovertemplate=GLOSSARY["s1"] + '<br>t=%{x:.2f}s<extra></extra>',
        ))
        fig2.add_trace(go.Scatter(
            x=s2_indices / sr, y=envelope[s2_indices], mode='markers', name='S2 (dub)',
            marker=dict(size=11, color='#a855f7', symbol='triangle-down'),
            hovertemplate=GLOSSARY["s2"] + '<br>t=%{x:.2f}s<extra></extra>',
        ))
        fig2.update_layout(title="Envelope Extraction & Heartbeat Peak Detection", xaxis_title="Time (seconds)", yaxis_title="Energy Envelope", template="plotly_dark", height=400)
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(
            "S1 (\"lub\") and S2 (\"dub\") are located heuristically from the envelope's local peaks within each "
            "detected cycle — not a validated clinical segmentation. When the two sounds blur together "
            "(common in quieter or murmur-heavy recordings), only S1 is shown for that cycle."
        )

    with tab3:
        st.subheader("📡 High-Frame-Rate Cardiac Sweep Monitor (60 FPS)")
        st.caption("Real-time oscilloscope sweep rendering synchronized with detected heartbeat intervals.")

        # Downsample signal for visual pacing (keeps sharpest sample per window so the
        # QRS spike survives instead of being averaged/strided away into noise)
        target_sample_rate = 250
        downsample_factor = max(1, int(sr / target_sample_rate))
        downsampled_ecg = downsample_preserve_peaks(simulated_ecg, downsample_factor)

        ecg_json = json.dumps(downsampled_ecg.tolist())
        bpm_val = float(bpm)

        # Canvas Sweep Monitor Component (White Trace + Green Border)
        canvas_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; background-color: #020802; overflow: hidden; }}
                .container {{
                    background-color: #050b05;
                    border: 2px solid #00ff41;
                    border-radius: 8px;
                    padding: 12px;
                    box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
                }}
                .header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    color: #00ff41;
                    font-family: monospace;
                    font-size: 14px;
                    margin-bottom: 8px;
                }}
                .beat-dot {{
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: #003300;
                    margin-left: 6px;
                    box-shadow: 0 0 4px rgba(0, 255, 65, 0.2);
                    transition: background 0.05s linear;
                }}
                #soundToggle {{
                    cursor: pointer;
                    user-select: none;
                    border: 1px solid #00ff41;
                    padding: 2px 10px;
                    border-radius: 4px;
                    color: #00ff41;
                }}
                #soundToggle:hover {{
                    background: rgba(0, 255, 65, 0.1);
                }}
                canvas {{
                    width: 100%;
                    height: 250px;
                    background-color: #020802;
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <span>LEAD II | 25mm/s</span>
                    <span>HR: <strong style="color: #ffffff; font-size: 18px;">{bpm_val:.0f}</strong> BPM<span id="beatDot" class="beat-dot"></span></span>
                    <span id="soundToggle">🔇 Sound Off</span>
                    <span>SWEEP: 60 FPS</span>
                </div>
                <canvas id="ecgCanvas" width="900" height="250"></canvas>
            </div>

            <script>
                const canvas = document.getElementById('ecgCanvas');
                const ctx = canvas.getContext('2d');

                const ecgData = {ecg_json};
                const totalSamples = ecgData.length;

                let cursorX = 0;
                let dataIndex = 0;
                const stepPx = 2;
                const fadeWidth = 45;

                // --- ECG "bip" audio: monitor-style beep synced to the R-wave peak ---
                const PEAK_THRESHOLD = 0.6;   // R-wave amplitude (~1.0) clears this; P/T waves (<=0.35) don't
                const RESET_THRESHOLD = 0.3;  // re-arm once the trace drops back toward baseline
                let peakArmed = true;
                let soundEnabled = false;
                let audioCtx = null;

                const soundToggle = document.getElementById('soundToggle');
                const beatDot = document.getElementById('beatDot');

                soundToggle.addEventListener('click', () => {{
                    if (!audioCtx) {{
                        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    }}
                    if (audioCtx.state === 'suspended') {{
                        audioCtx.resume();
                    }}
                    soundEnabled = !soundEnabled;
                    soundToggle.textContent = soundEnabled ? '🔊 Sound On' : '🔇 Sound Off';
                    soundToggle.style.background = soundEnabled ? 'rgba(0, 255, 65, 0.15)' : '';
                }});

                function playBeep() {{
                    if (!soundEnabled || !audioCtx) return;
                    const now = audioCtx.currentTime;
                    const osc = audioCtx.createOscillator();
                    const filter = audioCtx.createBiquadFilter();
                    const gain = audioCtx.createGain();

                    // Square wave -> clinical-monitor "pip" timbre (sine is too soft/musical);
                    // a lowpass tames the harsh upper harmonics without losing the sharp attack.
                    osc.type = 'square';
                    osc.frequency.value = 1000;
                    filter.type = 'lowpass';
                    filter.frequency.value = 2800;

                    // Near-instant attack, quick decay -> short percussive "pip" not a tone
                    gain.gain.setValueAtTime(0.0001, now);
                    gain.gain.exponentialRampToValueAtTime(0.25, now + 0.002);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.07);

                    osc.connect(filter);
                    filter.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.start(now);
                    osc.stop(now + 0.08);
                }}

                function flashBeat() {{
                    beatDot.style.background = '#00ff41';
                    beatDot.style.boxShadow = '0 0 8px 3px rgba(0, 255, 65, 0.9)';
                    setTimeout(() => {{
                        beatDot.style.background = '#003300';
                        beatDot.style.boxShadow = '0 0 4px rgba(0, 255, 65, 0.2)';
                    }}, 120);
                }}

                function drawGridSection(startX, endX) {{
                    ctx.strokeStyle = "rgba(0, 255, 65, 0.08)";
                    ctx.lineWidth = 1;

                    for (let x = startX; x < endX; x++) {{
                        if (x % 20 === 0 && x < canvas.width) {{
                            ctx.beginPath();
                            ctx.moveTo(x, 0);
                            ctx.lineTo(x, canvas.height);
                            ctx.stroke();
                        }}
                    }}
                }}

                // Draw initial background grid
                ctx.strokeStyle = "rgba(0, 255, 65, 0.08)";
                ctx.lineWidth = 1;
                for (let x = 0; x < canvas.width; x += 20) {{
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, canvas.height);
                    ctx.stroke();
                }}
                for (let y = 0; y < canvas.height; y += 20) {{
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(canvas.width, y);
                    ctx.stroke();
                }}

                function animate() {{
                    const height = canvas.height;
                    const width = canvas.width;
                    const midY = height / 2 + 10;
                    const scaleY = 75;

                    let nextX = cursorX + stepPx;
                    let wrapOccurred = false;

                    if (nextX >= width) {{
                        nextX = 0;
                        wrapOccurred = true;
                    }}

                    // Clear trailing space
                    ctx.fillStyle = "#020802";
                    ctx.fillRect(nextX, 0, fadeWidth, height);

                    // Restore grid
                    drawGridSection(nextX, nextX + fadeWidth);

                    let y1 = midY - (ecgData[dataIndex] * scaleY);
                    dataIndex = (dataIndex + 2) % totalSamples;
                    let y2 = midY - (ecgData[dataIndex] * scaleY);

                    // Fire the "bip" once per R-wave as the sweep crosses it
                    const currentVal = ecgData[dataIndex];
                    if (currentVal > PEAK_THRESHOLD && peakArmed) {{
                        peakArmed = false;
                        playBeep();
                        flashBeat();
                    }} else if (currentVal < RESET_THRESHOLD) {{
                        peakArmed = true;
                    }}

                    if (!wrapOccurred) {{
                        // White Trace Signal with Subtle Glow
                        ctx.shadowColor = '#ffffff';
                        ctx.shadowBlur = 6;
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 2.4;

                        ctx.beginPath();
                        ctx.moveTo(cursorX, y1);
                        ctx.lineTo(nextX, y2);
                        ctx.stroke();
                        ctx.shadowBlur = 0;
                    }}

                    cursorX = nextX;
                    requestAnimationFrame(animate);
                }}

                animate();
            </script>
        </body>
        </html>
        """

        st.components.v1.html(canvas_html, height=320)

    with tab4:
        st.subheader("🎯 Waveform Playback Cursor")
        st.caption("Press play — the orange line sweeps across the raw waveform in sync with the audio's actual playback position.")
        audio_bytes, mime = _get_audio_bytes_and_mime(audio_player_source, selected_filename)
        render_synced_waveform(time_ds_raw, signal_ds, audio_bytes, mime, key="single")

    # Condition Analysis Section
    st.subheader("📋 Condition Analysis")
    RISK_ALERT.get(status, st.warning)(f"**{status}**")

    for f in findings:
        st.markdown(f"- {f}")

    pdf_bytes = generate_pdf_report(
        dataset_name=selected_set,
        file_name=selected_filename,
        sample_rate=sr,
        duration_sec=len(signal) / sr,
        lowcut=lowcut,
        highcut=highcut,
        bpm=bpm,
        hrv=hrv,
        num_beats=len(peaks),
        interval_stats=interval_stats,
        status=status,
        findings=findings,
        checklist=checklist,
        disclaimer=EDUCATIONAL_DISCLAIMER,
        time=time,
        signal=signal,
        filtered_signal=filtered_signal,
        envelope=envelope,
        peaks=peaks,
        peak_times=peak_times,
        s1_indices=s1_indices,
        s2_indices=s2_indices,
    )
    st.download_button(
        label="🖨️ Print Full Report (Download PDF)",
        data=pdf_bytes,
        file_name=f"HeartByte_Report_{os.path.splitext(selected_filename)[0]}.pdf",
        mime="application/pdf",
        type="primary",
        key="download_single",
    )


def _render_compact_charts(result, key_prefix):
    signal, filtered_signal, envelope = result["signal"], result["filtered_signal"], result["envelope"]
    time = result["time"]
    sr = result["sr"]
    s1_indices, s2_indices = result["s1_indices"], result["s2_indices"]

    # Compare mode shows up to 4 of these charts at once, so keeping each trace
    # thinned (see _downsample_for_plot) matters even more here than in single mode.
    signal_ds, time_ds_raw = _downsample_for_plot(signal, time)
    filtered_ds, time_ds_filt = _downsample_for_plot(filtered_signal, time)
    envelope_ds, time_ds_env = _downsample_for_plot(envelope, time)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_ds_raw, y=signal_ds, mode='lines', name='Raw', opacity=0.4, line=dict(color='gray')))
    fig1.add_trace(go.Scatter(x=time_ds_filt, y=filtered_ds, mode='lines', name='Filtered', line=dict(color='orange'), yaxis='y2'))
    fig1.update_layout(
        title="Raw vs. Filtered", xaxis_title="Time (s)",
        yaxis=dict(title="Raw"), yaxis2=dict(title="Filtered", overlaying='y', side='right', showgrid=False),
        template="plotly_dark", height=300, margin=dict(t=35, b=30, l=45, r=45),
    )
    st.plotly_chart(fig1, use_container_width=True, key=f"fig1_{key_prefix}", config=PLOTLY_CONFIG)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_ds_env, y=envelope_ds, mode='lines', name='Envelope', line=dict(color='cyan', width=2)))
    fig2.add_trace(go.Scatter(
        x=s1_indices / sr, y=envelope[s1_indices], mode='markers', name='S1',
        marker=dict(size=9, color='#f97316', symbol='triangle-up'),
        hovertemplate=GLOSSARY["s1"] + '<br>t=%{x:.2f}s<extra></extra>',
    ))
    fig2.add_trace(go.Scatter(
        x=s2_indices / sr, y=envelope[s2_indices], mode='markers', name='S2',
        marker=dict(size=9, color='#a855f7', symbol='triangle-down'),
        hovertemplate=GLOSSARY["s2"] + '<br>t=%{x:.2f}s<extra></extra>',
    ))
    fig2.update_layout(
        title="Envelope & Peaks", xaxis_title="Time (s)", yaxis_title="Envelope",
        template="plotly_dark", height=300, margin=dict(t=35, b=30, l=45, r=45),
    )
    st.plotly_chart(fig2, use_container_width=True, key=f"fig2_{key_prefix}", config=PLOTLY_CONFIG)


def render_compare_mode(set_a, name_a, path_a, player_a, set_b, name_b, path_b, player_b, lowcut, highcut):
    result_a = run_analysis(path_a, name_a, lowcut, highcut)
    result_b = run_analysis(path_b, name_b, lowcut, highcut)

    st.subheader("Heartbeat Comparison")

    col_a, col_mid, col_b = st.columns([1, 0.85, 1])

    HEADING_STYLE = "white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:0.5rem;"

    with col_a:
        st.markdown(f"<h4 style='{HEADING_STYLE}' title='{name_a}'>🅰️ {name_a}</h4>", unsafe_allow_html=True)
        st.markdown(
            f"{pulsing_heart_html(result_a['bpm'], size='1.3rem')} &nbsp; **{result_a['bpm']:.1f} BPM**",
            unsafe_allow_html=True,
        )
        st.audio(player_a)
        _render_compact_charts(result_a, "a")

    with col_b:
        st.markdown(f"<h4 style='{HEADING_STYLE}' title='{name_b}'>🅱️ {name_b}</h4>", unsafe_allow_html=True)
        st.markdown(
            f"{pulsing_heart_html(result_b['bpm'], size='1.3rem')} &nbsp; **{result_b['bpm']:.1f} BPM**",
            unsafe_allow_html=True,
        )
        st.audio(player_b)
        _render_compact_charts(result_b, "b")

    with col_mid:
        st.markdown(f"<h4 style='{HEADING_STYLE}'>📊 Parameters</h4>", unsafe_allow_html=True)
        # Spacer to match the height of the pulsing-heart BPM line + audio player
        # rendered in the side columns, so all three columns line up vertically.
        st.markdown("<div style='height:90px;'></div>", unsafe_allow_html=True)

        rows = [
            ("Dataset", set_a, set_b, GLOSSARY["dataset"]),
            ("Sample Rate", f"{result_a['sr']} Hz", f"{result_b['sr']} Hz", GLOSSARY["sample_rate"]),
            ("Duration", f"{len(result_a['signal']) / result_a['sr']:.1f}s", f"{len(result_b['signal']) / result_b['sr']:.1f}s", GLOSSARY["duration"]),
            ("Bandpass Filter", f"{lowcut}–{highcut} Hz", f"{lowcut}–{highcut} Hz", GLOSSARY["bandpass_filter"]),
            ("Heart Rate", f"{result_a['bpm']:.1f} BPM", f"{result_b['bpm']:.1f} BPM", GLOSSARY["bpm"]),
            ("HRV", f"{result_a['hrv']:.1f} ms", f"{result_b['hrv']:.1f} ms", GLOSSARY["hrv"]),
            ("Detected Beats", str(len(result_a['peaks'])), str(len(result_b['peaks'])), GLOSSARY["beats"]),
            ("Mean Interval", f"{result_a['interval_stats']['mean_ms']:.0f} ms", f"{result_b['interval_stats']['mean_ms']:.0f} ms", GLOSSARY["mean_interval"]),
            (
                "Interval Range",
                f"{result_a['interval_stats']['min_ms']:.0f}–{result_a['interval_stats']['max_ms']:.0f} ms",
                f"{result_b['interval_stats']['min_ms']:.0f}–{result_b['interval_stats']['max_ms']:.0f} ms",
                GLOSSARY["interval_range"],
            ),
            ("Status", result_a['status'], result_b['status'], GLOSSARY["status"]),
        ]
        rows_html = "".join(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        gap:10px; padding:9px 2px; border-bottom:1px solid rgba(255,255,255,0.1);" title="{tooltip}">
                <span style="opacity:0.65; font-size:0.85rem;">{label}</span>
                <span style="text-align:right; font-size:0.85rem; line-height:1.5;">
                    <span style="color:#f97316">🅰️ {val_a}</span><br>
                    <span style="color:#38bdf8">🅱️ {val_b}</span>
                </span>
            </div>
            """
            for label, val_a, val_b, tooltip in rows
        )
        st.markdown(
            f"""
            <div style="border:1px solid rgba(255,255,255,0.15); border-radius:8px;
                        padding:4px 14px; height:600px; overflow-y:auto;">
                {rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("📋 Condition Analysis")
    cond_a, cond_b = st.columns(2)
    with cond_a:
        st.markdown(f"**Sample A — {name_a}**")
        RISK_ALERT.get(result_a["status"], st.warning)(f"**{result_a['status']}**")
        for f in result_a["findings"]:
            st.markdown(f"- {f}")
    with cond_b:
        st.markdown(f"**Sample B — {name_b}**")
        RISK_ALERT.get(result_b["status"], st.warning)(f"**{result_b['status']}**")
        for f in result_b["findings"]:
            st.markdown(f"- {f}")

    def _sample_payload(dataset_name, file_name, result):
        return dict(
            dataset_name=dataset_name, file_name=file_name, sample_rate=result["sr"],
            duration_sec=len(result["signal"]) / result["sr"], bpm=result["bpm"], hrv=result["hrv"],
            num_beats=len(result["peaks"]), interval_stats=result["interval_stats"], status=result["status"],
            findings=result["findings"], checklist=result["checklist"], time=result["time"],
            signal=result["signal"], filtered_signal=result["filtered_signal"], envelope=result["envelope"],
            peaks=result["peaks"], peak_times=result["peak_times"],
            s1_indices=result["s1_indices"], s2_indices=result["s2_indices"],
        )

    pdf_bytes = generate_comparison_pdf_report(
        sample_a=_sample_payload(set_a, name_a, result_a),
        sample_b=_sample_payload(set_b, name_b, result_b),
        lowcut=lowcut, highcut=highcut, disclaimer=EDUCATIONAL_DISCLAIMER,
    )
    st.download_button(
        label="🖨️ Print Comparison Report (Download PDF)",
        data=pdf_bytes,
        file_name=f"HeartByte_Comparison_{os.path.splitext(name_a)[0]}_vs_{os.path.splitext(name_b)[0]}.pdf",
        mime="application/pdf",
        type="primary",
        key="download_comparison",
    )


@st.cache_data
def _get_intro_heartbeat_audio_b64():
    """
    A short real heartbeat clip (trimmed from the archive), looped continuously
    from the moment the front-page heart is clicked through the pulsing beat
    and the home/menu screen, until one of the 4 menu options is chosen.
    Trimmed to start right at the first detected beat (not at the recording's
    t=0) so there's no lead-in silence/noise before the sound, and short
    enough that the loop reads as a natural, repeating heartbeat rhythm.

    Source file matters here: many recordings build up in volume over the
    first few beats, so a naive "trim to the first beat" still sounds weak
    until a later, louder beat arrives. This file was picked by scanning the
    archive for one where the very first beat is already at (99% of) the
    clip's peak loudness — its first six beats measure 0.2166/0.2147/0.2158/
    0.2168/0.2130/0.2198, essentially flat — so the very first sound you hear
    is already the "real", full-strength heartbeat, not a quiet lead-in.
    """
    path = os.path.join("archive", "set_a", "normal__201108011115.wav")
    signal, sr = load_audio_file(path, target_duration=5.0)
    filtered = bandpass_filter(signal, lowcut=20, highcut=300, fs=sr)
    envelope = extract_envelope(filtered, fs=sr)
    peaks, _, _, _ = analyze_peaks(envelope, fs=sr)

    lead_in = int(0.03 * sr)  # ~30ms of natural attack before the beat, not a hard cut
    start = max(0, int(peaks[0]) - lead_in) if len(peaks) else 0
    clip_len = int(1.3 * sr)  # one loop cycle: roughly one lub-dub + a short diastole pause
    clip = signal[start:start + clip_len].astype(np.float64)

    # Raw stethoscope recordings sit well below full scale, and the browser's
    # own <audio>.volume already maxes out at 1.0 — so the only way to make
    # this actually louder is to boost the samples themselves. Peak-normalize,
    # then push it through a soft (tanh) saturation curve: that raises the
    # quieter diastole portion along with the loud lub-dub peaks, which reads
    # as noticeably louder than a plain peak-normalize while staying free of
    # harsh digital clipping.
    peak = np.max(np.abs(clip)) if clip.size else 0.0
    if peak > 1e-6:
        normalized = clip / peak
        drive = 12.0  # ~+9.5dB RMS over the raw clip — clearly louder, no hard clipping (peak stays ~0.97)
        clip = (np.tanh(normalized * drive) / np.tanh(drive)) * 0.97

    buf = io.BytesIO()
    sf.write(buf, clip.astype(np.float32), sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_home():
    # No-scroll, single-viewport magenta backdrop shared by every stage below
    # (idle "front page", pulsing beat, and the menu "home page").
    _inject_home_theme()

    # Scoped to the home page only (removed the moment view != "home", since this
    # markdown call simply won't run): centers the global st.title() heading and
    # bumps it up slightly while keeping it on one line (nowrap already forces
    # that; the clamp() just needs a touch more headroom than the default).
    st.markdown(
        """
        <style>
        h1 {
            text-align: center !important;
            font-size: clamp(1rem, 2.3vw, 2.1rem) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center; margin:-0.4rem 0 0.9rem 0;'>"
        "<span class='hb-credit'>✦ Made by Team Discrete_Minds ✦</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    stage = st.session_state.home_stage

    # All three stages render into this single placeholder instead of directly
    # into the page. Streamlit only sweeps away a run's *stale trailing*
    # elements once the whole script finishes — but the "pulsing" stage below
    # calls time.sleep() mid-script, so without a shared container the "idle"
    # stage's heart button/caption stayed on screen, un-swept, for the entire
    # sleep, alongside the new pulsing iframe. Reusing one st.empty() container
    # makes each stage switch replace its content immediately, sleep or not.
    stage_slot = st.empty()

    with stage_slot.container():
        _render_home_stage(stage)


def _render_home_stage(stage):
    if stage == "idle":
        st.markdown("<div style='height:13vh;'></div>", unsafe_allow_html=True)
        # Scoped to this branch only: the heart button is the only button on
        # the page here, so it's safe to style *all* buttons this large. Drawn
        # as a glossy magenta 3D orb (radial-gradient highlight + inset shadows)
        # holding the heart emoji, with a slow pulsing box-shadow for a "dynamic
        # 3D" idle feel; only box-shadow animates (never transform), so the
        # hover/active scale below never fights a keyframe for the same property.
        st.markdown(
            """
            <style>
            div[data-testid="stButton"] {
                display: flex; justify-content: center;
                animation: fadeInUp 0.6s ease-out both;
                position: relative; z-index: 1;
            }
            @keyframes heartGlow3D {
                0%, 100% {
                    box-shadow: 0 25px 60px rgba(122,14,110,0.55),
                                inset -14px -14px 34px rgba(26,11,46,0.35),
                                inset 10px 10px 26px rgba(255,255,255,0.22);
                }
                50% {
                    box-shadow: 0 32px 80px rgba(255,45,142,0.7),
                                inset -18px -18px 40px rgba(26,11,46,0.4),
                                inset 12px 12px 30px rgba(255,255,255,0.32);
                }
            }
            div[data-testid="stButton"] button {
                font-size: 7.4rem !important; line-height: 1.3 !important;
                height: 245px !important; width: 245px !important;
                min-height: 245px !important; padding: 0 !important;
                border-radius: 50% !important;
                background: radial-gradient(circle at 32% 26%, rgba(255,255,255,0.68) 0%,
                            rgba(255,228,242,0.55) 14%, rgba(255,91,168,0.55) 38%,
                            rgba(214,20,130,0.55) 65%, rgba(90,10,60,0.5) 100%) !important;
                border: 1px solid rgba(255,228,242,0.4) !important; overflow: visible !important;
                display: flex !important; align-items: center !important; justify-content: center !important;
                cursor: pointer !important;
                animation: heartGlow3D 3.2s ease-in-out infinite !important;
                transition: transform 0.25s ease !important;
            }
            div[data-testid="stButton"] button:hover { transform: scale(1.08) !important; }
            div[data-testid="stButton"] button:active { transform: scale(0.94) !important; }
            div[data-testid="stButton"] button p {
                font-size: 7.4rem !important; line-height: 1 !important; margin: 0 !important;
                filter: drop-shadow(0 10px 18px rgba(176,17,107,0.55)) drop-shadow(0 0 30px rgba(255,91,168,0.4));
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        _, heart_col, _ = st.columns([2, 1, 2])
        with heart_col:
            st.button("🫀", key="home_heart_btn", on_click=lambda: st.session_state.update(home_stage="pulsing"))
        st.markdown(
            """
            <p style="text-align:center; margin-top:1.2rem; font-style:italic; font-weight:600;
                      font-size:1.05rem; color:#ffd6ec; text-transform:uppercase;
                      animation: breathe 2s ease-in-out infinite, fadeInUp 0.6s ease-out 0.1s both;">
                Click the heart to begin
            </p>
            """,
            unsafe_allow_html=True,
        )

        # Makes the beat sound actually audible. A fresh iframe created *after*
        # the click (like the old "pulsing" stage's own <audio> tag below used
        # to be) never inherits the click's user gesture, so browsers silently
        # block/mute its autoplay — that was the "click heart, hear nothing" bug.
        # The fix: attach a real addEventListener('click', ...) directly on the
        # heart button itself (this DOES fire as part of the genuine user
        # gesture), and build the Audio object against window.parent (the top
        # page), not this throwaway iframe — so playback (a) counts as
        # gesture-triggered and is allowed at full volume, and (b) keeps
        # playing right through the Streamlit rerun that swaps this iframe out
        # a moment later. It loops and is only paused when one of the 4 menu
        # options is chosen (see the "menu" stage below), so the same beat
        # carries through the pulsing screen and the home/menu screen.
        audio_b64 = _get_intro_heartbeat_audio_b64()
        st.components.v1.html(
            f"""
            <script>
            (function() {{
                const doc = window.parent.document;
                const btn = doc.querySelector('div[data-testid="stButton"] button');
                if (btn && !btn.__hbBound) {{
                    btn.__hbBound = true;
                    btn.addEventListener('click', function() {{
                        try {{
                            const audio = new window.parent.Audio('data:audio/wav;base64,{audio_b64}');
                            audio.loop = true;
                            audio.volume = 1.0;
                            window.parent.__hbAudio = audio;
                            audio.play().catch(() => {{}});
                        }} catch (e) {{}}
                    }});
                }}
            }})();
            </script>
            """,
            height=0,
        )

    elif stage == "pulsing":
        # Purely visual now — the beat sound itself was already started by the
        # idle stage's click listener (on window.parent, so it survives this
        # stage swap and keeps playing right through this screen). This iframe
        # only shows the pulsing orb and auto-advances to the menu.
        #
        # No server-side time.sleep() here on purpose: Streamlit only sweeps away a
        # run's stale/removed elements once the *whole script run finishes* — pausing
        # mid-run with time.sleep() left this stage's "Click the heart to begin" button
        # visible on screen (unswept) for the entire pause, alongside this iframe.
        # Instead, this run finishes immediately (so the previous stage's elements are
        # cleaned up right away), and the iframe's own JS below auto-advances to the
        # "menu" stage after the same delay by clicking a hidden Streamlit button —
        # whose on_click callback flips the stage and triggers a fresh, equally quick
        # rerun (so nothing lingers on that transition either).
        st.components.v1.html(
            """
            <!DOCTYPE html>
            <html>
            <head>
            <style>
                body { margin: 0; background: transparent; font-family: sans-serif; }
                @keyframes heartPulse {
                    0%   { transform: scale(1); }
                    15%  { transform: scale(1.28); }
                    30%  { transform: scale(1); }
                    45%  { transform: scale(1.15); }
                    60%  { transform: scale(1); }
                    100% { transform: scale(1); }
                }
                .orb {
                    width: 210px; height: 210px; margin: 0 auto; border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    background: radial-gradient(circle at 32% 26%, rgba(255,255,255,0.68) 0%,
                                rgba(255,228,242,0.55) 14%, rgba(255,91,168,0.55) 38%,
                                rgba(214,20,130,0.55) 65%, rgba(90,10,60,0.5) 100%);
                    border: 1px solid rgba(255,228,242,0.4);
                    box-shadow: 0 25px 60px rgba(122,14,110,0.55),
                                inset -14px -14px 34px rgba(26,11,46,0.35),
                                inset 10px 10px 26px rgba(255,255,255,0.22);
                    animation: heartPulse 0.5s ease-in-out infinite;
                }
                .heart {
                    display: block; text-align: center; font-size: 6.2rem;
                    filter: drop-shadow(0 10px 18px rgba(176,17,107,0.55)) drop-shadow(0 0 30px rgba(255,91,168,0.4));
                }
                .caption { text-align: center; opacity: 0.85; color: #ffd6ec; margin-top: 16px; font-size: 0.95rem; letter-spacing: 0.5px; }
            </style>
            </head>
            <body>
                <div style="padding-top:20px;">
                    <div class="orb"><span class="heart">🫀</span></div>
                    <p class="caption">Starting HeartByte…</p>
                </div>
                <script>
                    // Auto-advance to the menu after the intro beat: this iframe is
                    // same-origin (Streamlit's sandbox allows it), so it can reach into
                    // the parent page and click the hidden advance button below. Timed
                    // to land 1.3s after the click, inside the requested 1-1.5s window.
                    setTimeout(function() {
                        try {
                            const btn = window.parent.document.querySelector(
                                'div[data-testid="stButton"] button[kind="secondary"]'
                            ) || window.parent.document.querySelector('div[data-testid="stButton"] button');
                            if (btn) { btn.click(); }
                        } catch (e) {}
                    }, 1300);
                </script>
            </body>
            </html>
            """,
            height=290,
        )

        # Hidden trigger clicked by the iframe's JS above once the intro pulse has
        # played out. Safe to style *all* buttons here: this is the only stButton
        # rendered during the "pulsing" stage.
        st.markdown(
            """
            <style>
            div[data-testid="stButton"] {
                position: fixed !important; left: -9999px !important; top: -9999px !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.button("Continue", key="hb_auto_advance", on_click=lambda: st.session_state.update(home_stage="menu"))

    elif stage == "menu":
        st.markdown(
            f"<div style='text-align:center;'>{pulsing_heart_html(70, size='3rem')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # Scoped to this branch: the heading and buttons here are the only ones on
        # the page, so a broad selector is safe — gives the menu a staggered
        # fade/slide-in as it emerges after the intro pulse, and restyles the
        # card + buttons as frosted magenta glass to match the front page (hover
        # picks up the cyan ring accent for contrast).
        st.markdown(
            """
            <style>
            div[data-testid="stHeadingWithActionElements"] {
                animation: fadeInUp 0.45s ease-out both;
            }
            div[data-testid="stButton"] {
                animation: fadeInUp 0.45s ease-out both;
                animation-delay: 0.15s;
            }
            /* st.container(key="feature_card") below adds this .st-key-feature_card
               class to its own stVerticalBlock div, which is the one that actually
               carries the border — a more version-stable hook than data-testid,
               since which wrapper element renders the border differs across
               Streamlit releases. */
            .st-key-feature_card {
                background: rgba(26,11,46,0.42) !important;
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255,180,225,0.3) !important;
                border-radius: 20px !important;
                box-shadow: 0 25px 60px rgba(26,11,46,0.5) !important;
            }
            .st-key-feature_card h3 { color: #f9ecf5 !important; }
            .st-key-feature_card button {
                border-color: rgba(255,180,225,0.35) !important;
                color: #f9ecf5 !important;
                background: rgba(255,45,142,0.1) !important;
            }
            .st-key-feature_card button:hover {
                border-color: #7cf5ff !important;
                color: #ffffff !important;
                background: rgba(124,245,255,0.14) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        _, mid, _ = st.columns([1, 1.4, 1])
        with mid:
            with st.container(border=True, key="feature_card"):
                st.markdown(
                    "<h3 style='text-align:center; margin-top:0;'>Choose a Feature</h3>",
                    unsafe_allow_html=True,
                )
                if st.button("🔍 Single Sample Analysis", use_container_width=True, key="nav_single"):
                    st.session_state.view = "single"
                    st.rerun()
                if st.button("⚖️ Compare Two Samples", use_container_width=True, key="nav_compare"):
                    st.session_state.view = "compare"
                    st.rerun()
                if st.button("ℹ️ About Us", use_container_width=True, key="nav_about"):
                    st.session_state.view = "about"
                    st.rerun()
                if st.button("🚪 Exit", use_container_width=True, key="nav_exit"):
                    st.session_state.view = "exiting"
                    st.rerun()

        # The looping beat sound started by the idle stage's click listener
        # keeps playing (on window.parent) straight through this menu screen;
        # stop it the moment any of the 4 options above is chosen, using the
        # same native-click-listener trick so it fires in the same gesture as
        # Streamlit's own on_click, before the rerun navigates away.
        st.components.v1.html(
            """
            <script>
            (function() {
                const doc = window.parent.document;
                const card = doc.querySelector('.st-key-feature_card');
                if (!card) return;
                card.querySelectorAll('button').forEach(function(btn) {
                    if (!btn.__hbStopBound) {
                        btn.__hbStopBound = true;
                        btn.addEventListener('click', function() {
                            try {
                                if (window.parent.__hbAudio) { window.parent.__hbAudio.pause(); }
                            } catch (e) {}
                        });
                    }
                });
            })();
            </script>
            """,
            height=0,
        )


def render_back_button():
    if st.sidebar.button("⬅️ Back to Home", key="back_home", use_container_width=True):
        st.session_state.view = "home"
        st.session_state.home_stage = "menu"
        st.rerun()
    st.sidebar.markdown("---")


def render_about_page():
    st.header("ℹ️ About HeartByte")

    st.markdown(
        "Every heartbeat carries a story — a rhythm of valves closing, blood flowing, and a body doing its "
        "quiet, relentless work. HeartByte turns that story into something you can see: pick a heartbeat "
        "recording and watch raw sound become a filtered waveform, an energy envelope, individually labeled "
        "S1 (\"lub\") and S2 (\"dub\") heart sounds, and a synchronized simulated ECG trace."
    )

    st.markdown(
        "This project is made by Team Discrete_Minds: Dibya Joti Kundu (ID-2305099) and Avijid Roy Himel "
        "(ID-2305100) as the term project for CSE-220 (Signal and Linear Systems Sessional) under the "
        "supervision of Al Muhit Muhtadi Sir."
    )

    st.markdown(
        "Under the hood, HeartByte applies digital signal processing — Butterworth bandpass filtering, "
        "envelope extraction, and peak detection — to isolate and analyze heart sounds (PCG recordings), and "
        "uses a trained classifier to confirm an upload is a genuine heartbeat before analyzing it. It supports "
        "single-recording analysis, side-by-side comparison of two recordings, and exportable PDF reports. "
        "It's built as an educational demonstration of signal processing concepts — not a certified medical "
        "diagnostic tool."
    )


def render_exit():
    # Once the process actually dies below, Streamlit's frontend detects the
    # dropped WebSocket and shows its own "Connection error" dialog + status
    # widget — normally alarming/technical-looking for an intentional exit, and
    # since we don't want that, we hide it.
    #
    # os._exit(0) below kills the process immediately, mid-script, so Streamlit
    # never reaches the normal "script finished" signal that tells the frontend
    # to sweep away the previous run's stale elements (the menu screen this
    # replaced). Rather than leaving that menu visible underneath, this renders
    # a fixed, full-viewport overlay (in the *parent* page, not an iframe) that
    # simply paints over everything else regardless of what's left behind it.
    # The "Closing…" -> "closed" text swap is done in pure CSS (two stacked
    # layers cross-faded via a delayed keyframe) so it doesn't depend on script
    # execution inside unsafe_allow_html markdown, which Streamlit doesn't run.
    # A tiny hidden iframe alongside it is only there for the one thing that
    # truly needs JS: the timed window.close() attempt.
    st.markdown(
        """
        <style>
        [data-testid="stDialog"], [data-testid="stStatusWidget"],
        [data-testid="stToolbar"], [data-testid="stAppHeader"] {
            display: none !important;
        }
        html, body { overflow: hidden !important; }

        .hb-exit-overlay {
            position: fixed; inset: 0; z-index: 999999;
            display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #1a0b2e 0%, #7a0e6e 45%, #ff2d8e 100%);
        }
        .hb-exit-text { display: grid; text-align: center; color: #f9ecf5; font-family: sans-serif; }
        .hb-exit-text > div { grid-area: 1 / 1; }
        .hb-exit-text h2 { margin: 0 0 0.3rem 0; }
        .hb-exit-text p { margin: 0; opacity: 0.75; }
        .hb-exit-stage1 { animation: hbExitOut 0.01s linear 0.9s forwards; }
        .hb-exit-stage2 { opacity: 0; animation: hbExitIn 0.01s linear 0.9s forwards; }
        @keyframes hbExitOut { to { opacity: 0; visibility: hidden; } }
        @keyframes hbExitIn { to { opacity: 1; visibility: visible; } }
        </style>
        <div class="hb-exit-overlay">
            <div class="hb-exit-text">
                <div class="hb-exit-stage1">
                    <h2>👋 Thank you for using HeartByte!</h2>
                    <p>Closing…</p>
                </div>
                <div class="hb-exit-stage2">
                    <h2>✅ HeartByte has closed</h2>
                    <p>You can close this browser tab now.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.components.v1.html(
        """<script>setTimeout(() => { try { window.close(); } catch (e) {} }, 900);</script>""",
        height=0,
    )
    time.sleep(1.0)
    os._exit(0)


# View Dispatch
view = st.session_state.view

if view == "home":
    render_home()

elif view == "exiting":
    render_exit()

elif view == "about":
    render_back_button()
    render_about_page()

elif view == "single":
    render_back_button()
    st.sidebar.header("1. Select Audio Dataset")
    selection = select_audio_source("single")

    st.sidebar.header("2. Filter Parameters")
    lowcut = st.sidebar.slider("Bandpass Low Cutoff (Hz)", 10, 50, 20, key="lowcut_slider", help=GLOSSARY["lowcut"])
    highcut = st.sidebar.slider("Bandpass High Cutoff (Hz)", 100, 500, 300, key="highcut_slider", help=GLOSSARY["highcut"])

    if selection is None:
        st.info("👈 Select a heartbeat recording from the sidebar (archive or your own upload) to begin.")
        st.stop()
    selected_set, selected_filename, selected_filepath, audio_player_source = selection
    render_single_mode(selected_set, selected_filename, selected_filepath, audio_player_source, lowcut, highcut)

elif view == "compare":
    render_back_button()
    st.sidebar.header("1. Select Audio Dataset")
    st.sidebar.markdown("**🅰️ Heartbeat Sample A**")
    selection_a = select_audio_source("cmp_a", "Upload Sample A Recording")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🅱️ Heartbeat Sample B**")
    selection_b = select_audio_source("cmp_b", "Upload Sample B Recording")

    st.sidebar.header("2. Filter Parameters")
    lowcut = st.sidebar.slider("Bandpass Low Cutoff (Hz)", 10, 50, 20, key="lowcut_slider", help=GLOSSARY["lowcut"])
    highcut = st.sidebar.slider("Bandpass High Cutoff (Hz)", 100, 500, 300, key="highcut_slider", help=GLOSSARY["highcut"])

    if selection_a is None or selection_b is None:
        st.info("👈 Select both Sample A and Sample B heartbeat recordings from the sidebar to begin comparing.")
        st.stop()
    set_a, name_a, path_a, player_a = selection_a
    set_b, name_b, path_b, player_b = selection_b
    render_compare_mode(set_a, name_a, path_a, player_a, set_b, name_b, path_b, player_b, lowcut, highcut)
