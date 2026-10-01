import os
import io
import re
import json
import time
import base64
import mimetypes
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw


# ============================================================
# Streamlit page
# ============================================================

st.set_page_config(
    page_title="AI Pitching Analyzer",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ AI Pitching Analyzer")
st.caption(
    "Gemini API 기반 멀티모달 투구 분석 · "
    "영상 분석 → 릴리스 후보 탐지 → OpenCV 프레임 세분화 → 이미지 재분석"
)


# ============================================================
# Gemini config
# ============================================================

BASE_URL = "https://generativelanguage.googleapis.com"


def get_secret(name, default=None):
    """Read Streamlit secret first, then environment variable."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


API_KEY = get_secret("GEMINI_API_KEY")
MODEL = get_secret("MODEL", "gemini-3.1-flash-lite")

if not API_KEY:
    st.error(
        "GEMINI_API_KEY가 설정되어 있지 않습니다. "
        "Streamlit Community Cloud의 App settings → Secrets에 API 키를 등록하세요."
    )
    st.code('GEMINI_API_KEY = "YOUR_KEY"\nMODEL = "gemini-3.1-flash-lite"', language="toml")
    st.stop()


# ============================================================
# REST helpers
# ============================================================

def api_error(response):
    try:
        data = response.json()
        return data.get("error", {}).get("message", response.text)
    except Exception:
        return response.text


def request_with_retry(method, url, *, retries=3, **kwargs):
    retry_codes = {429, 500, 502, 503, 504}
    last_response = None

    for attempt in range(retries):
        try:
            response = requests.request(method, url, **kwargs)
            last_response = response
        except requests.exceptions.Timeout:
            if attempt == retries - 1:
                raise RuntimeError("Gemini 서버 응답 시간이 초과되었습니다.")
            time.sleep(2 ** attempt)
            continue

        if response.status_code not in retry_codes or attempt == retries - 1:
            return response

        time.sleep(2 ** attempt)

    return last_response


def call_gemini(parts):
    url = f"{BASE_URL}/v1beta/models/{MODEL}:generateContent"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ]
    }

    response = request_with_retry(
        "POST",
        url,
        headers={
            "x-goog-api-key": API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=(20, 180),
    )

    if not response.ok:
        raise RuntimeError(
            f"Gemini API 오류 {response.status_code}: {api_error(response)}"
        )

    data = response.json()
    candidates = data.get("candidates", [])

    if not candidates:
        raise RuntimeError("Gemini가 응답 후보를 반환하지 않았습니다.")

    output_parts = candidates[0].get("content", {}).get("parts", [])
    texts = [p["text"] for p in output_parts if "text" in p]

    if not texts:
        raise RuntimeError("Gemini 응답에 text가 없습니다.")

    return "\n".join(texts)


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass

    raise RuntimeError("Gemini 결과를 JSON으로 읽을 수 없습니다.\n\n" + text[:1800])


# ============================================================
# Gemini Files API
# ============================================================

def upload_video_to_gemini(video_path, mime_type, display_name):
    file_size = os.path.getsize(video_path)

    start_response = request_with_retry(
        "POST",
        f"{BASE_URL}/upload/v1beta/files",
        headers={
            "x-goog-api-key": API_KEY,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": display_name}},
        timeout=60,
    )

    if not start_response.ok:
        raise RuntimeError(
            f"영상 업로드 준비 실패 {start_response.status_code}: "
            f"{api_error(start_response)}"
        )

    upload_url = (
        start_response.headers.get("X-Goog-Upload-URL")
        or start_response.headers.get("x-goog-upload-url")
    )

    if not upload_url:
        raise RuntimeError("Gemini upload URL을 받지 못했습니다.")

    with open(video_path, "rb") as f:
        upload_response = request_with_retry(
            "POST",
            upload_url,
            headers={
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            data=f,
            timeout=(30, 300),
        )

    if not upload_response.ok:
        raise RuntimeError(
            f"영상 업로드 실패 {upload_response.status_code}: "
            f"{api_error(upload_response)}"
        )

    data = upload_response.json()
    return data.get("file", data)


def wait_until_active(file_info, timeout_seconds=300):
    started = time.time()

    while time.time() - started < timeout_seconds:
        state = file_info.get("state", "")

        if isinstance(state, dict):
            state = state.get("name", "")

        if state == "ACTIVE":
            return file_info

        if state == "FAILED":
            raise RuntimeError("Gemini의 영상 처리에 실패했습니다.")

        response = request_with_retry(
            "GET",
            f"{BASE_URL}/v1beta/{file_info['name']}",
            headers={"x-goog-api-key": API_KEY},
            timeout=30,
        )

        if not response.ok:
            raise RuntimeError(
                f"영상 상태 확인 실패 {response.status_code}: {api_error(response)}"
            )

        file_info = response.json()
        time.sleep(2)

    raise TimeoutError("Gemini 영상 처리 시간이 5분을 초과했습니다.")


# ============================================================
# Video analysis
# ============================================================

VIDEO_PROMPT = """
첨부된 야구 투구 영상을 분석하십시오.

영상에서 직접 확인되는 정보만 사용하고, 보이지 않는 내용을
일반적인 야구 지식으로 추측하지 마십시오.

투구 동작을 다음 6단계로 구분하십시오.
1. 준비 자세
2. 다리 움직임
3. 체중 이동
4. 팔 움직임
5. 릴리스
6. 팔로스루

각 단계의 시작/종료 시점(영상 시작 기준 초),
직접 관찰한 내용, 판단하기 어려운 내용을 기록하십시오.

특히 공이 투수의 손을 떠나는 것으로 보이는 대략적인 순간을
coarse_release_time_seconds에 기록하십시오.
불확실한 경우 release_confidence를 low로 설정하십시오.

정확한 관절 각도, 부상 위험, 선수의 심리 상태를 추측하지 마십시오.

반드시 JSON 하나만 출력하십시오.

{
  "summary": "전체 투구 동작 요약",
  "stages": [
    {
      "stage": "준비 자세",
      "start": 0.0,
      "end": 0.5,
      "observation": "영상에서 직접 관찰된 내용",
      "uncertainty": "판단하기 어려운 내용"
    }
  ],
  "coarse_release_time_seconds": 2.5,
  "release_confidence": "high 또는 medium 또는 low",
  "release_reason": "해당 시점을 릴리스로 판단한 이유",
  "limitations": ["영상 분석의 한계"]
}
""".strip()


def analyze_video_gemini(file_info):
    parts = [
        {
            "file_data": {
                "mime_type": file_info.get("mimeType", "video/mp4"),
                "file_uri": file_info["uri"],
            }
        },
        {"text": VIDEO_PROMPT},
    ]
    return parse_json(call_gemini(parts))


# ============================================================
# OpenCV
# ============================================================

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("OpenCV에서 영상을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not fps or fps <= 0:
        fps = 30

    duration = frames / fps if frames else 0

    return {
        "fps": float(fps),
        "frames": int(frames),
        "duration": float(duration),
        "width": width,
        "height": height,
    }


def extract_frame(video_path, seconds):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(seconds) * 1000)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"{seconds:.3f}초 프레임을 추출하지 못했습니다.")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def create_contact_sheet(video_path, release_time, duration):
    # 1차 릴리스 시점 ±0.7초를 15개 정지 프레임으로 재분석
    start = max(0.0, release_time - 0.7)
    end = min(duration, release_time + 0.7)
    times = np.linspace(start, end, 15)

    tile_w = 320
    tile_h = 240
    columns = 5
    tiles = []

    for i, t in enumerate(times, start=1):
        frame = extract_frame(video_path, float(t))
        frame.thumbnail((tile_w - 10, tile_h - 55))

        tile = Image.new("RGB", (tile_w, tile_h), "white")
        x = (tile_w - frame.width) // 2
        tile.paste(frame, (x, 48))

        draw = ImageDraw.Draw(tile)
        draw.text(
            (10, 12),
            f"Frame {i:02d} | {float(t):.3f} sec",
            fill="black",
        )
        tiles.append(tile)

    rows = int(np.ceil(len(tiles) / columns))
    sheet = Image.new(
        "RGB",
        (columns * tile_w, rows * tile_h),
        "white",
    )

    for i, tile in enumerate(tiles):
        sheet.paste(
            tile,
            ((i % columns) * tile_w, (i // columns) * tile_h),
        )

    return sheet, [float(x) for x in times]


def image_to_part(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": encoded,
        }
    }


# ============================================================
# Fine release-frame analysis
# ============================================================

def refine_release(sheet, times):
    timestamp_text = "\n".join(
        f"Frame {i + 1:02d}: {t:.3f} sec"
        for i, t in enumerate(times)
    )

    prompt = f"""
첨부 이미지는 한 번의 야구 투구에서 릴리스 예상 시점 주변을
연속적으로 추출한 15개의 후보 프레임입니다.

Timestamp:
{timestamp_text}

공이 손을 떠나는 순간에 가장 가까운 프레임 하나를 선택하십시오.

공과 손의 상대적 위치, 공의 분리 전/후 여부, 투구 팔 위치,
앞다리 상태, 상체 진행 방향을 관찰하십시오.
공이 잘 보이지 않는 경우 보인다고 추측하지 마십시오.

반드시 JSON 하나만 출력하십시오.

{{
  "selected_frame_number": 8,
  "confidence": "high 또는 medium 또는 low",
  "reason": "선택 이유",
  "observations": ["직접 관찰 내용"],
  "limitations": ["판단 한계"]
}}
""".strip()

    return parse_json(
        call_gemini(
            [
                image_to_part(sheet),
                {"text": prompt},
            ]
        )
    )


def cross_validate(final_image, timestamp, video_result):
    prompt = f"""
첨부 이미지는 앞선 영상 분석 후 선택된 최종 릴리스 후보 프레임입니다.

Timestamp: {timestamp:.3f} sec

앞선 영상 분석의 릴리스 판단:
{video_result.get("release_reason", "")}

현재 이미지를 직접 관찰하여 앞선 영상 분석과 비교하십시오.
보이지 않는 것은 추측하지 마십시오.

반드시 JSON 하나만 출력하십시오.

{{
  "release_state": "before_release 또는 release 또는 after_release 또는 uncertain",
  "direct_observations": ["이미지에서 직접 확인 가능한 사실"],
  "matches": ["영상 분석과 일치하는 내용"],
  "differences": ["영상 분석과 일치하지 않는 내용"],
  "cannot_determine": ["판단하기 어려운 내용"]
}}
""".strip()

    return parse_json(
        call_gemini(
            [
                image_to_part(final_image),
                {"text": prompt},
            ]
        )
    )


# ============================================================
# Report
# ============================================================

def bullets(items):
    if not items:
        return "- 없음"
    return "\n".join(f"- {item}" for item in items)


def make_stage_table(stages):
    rows = [
        "| 단계 | 시간 | 직접 관찰 내용 | 불확실성 |",
        "|---|---|---|---|",
    ]

    for stage in stages:
        name = str(stage.get("stage", "-")).replace("|", "/")
        start = stage.get("start", "-")
        end = stage.get("end", "-")
        obs = str(stage.get("observation", "-")).replace("|", "/").replace("\n", " ")
        unc = str(stage.get("uncertainty", "-")).replace("|", "/").replace("\n", " ")
        rows.append(f"| {name} | {start}–{end} s | {obs} | {unc} |")

    return "\n".join(rows)


def make_report(info, video_result, frame_result, refined_time, cross_result):
    return f"""# ⚾ AI Pitching Analyzer Report

**Gemini model:** `{MODEL}`

## 1. 영상 정보
- 영상 길이: **{info['duration']:.2f}초**
- FPS: **{info['fps']:.2f}**
- 해상도: **{info['width']} × {info['height']}**

## 2. 투구 전체 분석
{video_result.get('summary', '-')}

{make_stage_table(video_result.get('stages', []))}

## 3. 영상 기반 1차 릴리스 분석
- 1차 릴리스 추정: **{video_result.get('coarse_release_time_seconds', '-')}초**
- 신뢰도: **{video_result.get('release_confidence', '-')}**
- 판단 근거: {video_result.get('release_reason', '-')}

## 4. OpenCV + 이미지 기반 정밀 분석
- 선택 프레임: **Frame {frame_result.get('selected_frame_number', '-')}**
- 정밀 릴리스 추정: **{refined_time:.3f}초**
- 이미지 분석 신뢰도: **{frame_result.get('confidence', '-')}**
- 선택 이유: {frame_result.get('reason', '-')}

### 후보 프레임 관찰 내용
{bullets(frame_result.get('observations', []))}

### 후보 프레임 분석의 한계
{bullets(frame_result.get('limitations', []))}

## 5. 영상 ↔ 이미지 교차검증
**최종 프레임 상태:** `{cross_result.get('release_state', '-')}`

### 이미지에서 직접 확인
{bullets(cross_result.get('direct_observations', []))}

### 영상 분석과 일치
{bullets(cross_result.get('matches', []))}

### 영상 분석과 차이
{bullets(cross_result.get('differences', []))}

### 판단 불가능
{bullets(cross_result.get('cannot_determine', []))}

## 6. AI 분석의 한계
{bullets(video_result.get('limitations', []))}

---
본 결과는 AI 기반 영상 관찰 결과이며 전문적인 생체역학 측정이나 의학적 진단을 대체하지 않습니다.
"""


# ============================================================
# UI
# ============================================================

with st.sidebar:
    st.header("설정")
    st.write(f"**Gemini model**  \n`{MODEL}`")
    st.info(
        "수업 시연용으로는 투구 한 번만 포함된 "
        "3~10초 정도의 짧은 MP4 영상을 권장합니다."
    )
    st.caption(
        "API Key는 코드에 저장하지 않고 Streamlit Secrets에서 불러옵니다."
    )

uploaded_video = st.file_uploader(
    "📹 투구 영상을 업로드하세요",
    type=["mp4", "mov", "avi", "mpeg", "mpg", "webm", "wmv"],
)

if uploaded_video is not None:
    st.video(uploaded_video)

    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_clicked = st.button(
            "⚾ 분석 시작",
            type="primary",
            use_container_width=True,
        )
    with col2:
        st.caption(
            f"{uploaded_video.name} · {uploaded_video.size / 1024 / 1024:.1f} MB"
        )

    if analyze_clicked:
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        mime_type = (
            uploaded_video.type
            or mimetypes.guess_type(uploaded_video.name)[0]
            or "video/mp4"
        )

        temp_video = None

        try:
            progress = st.progress(0, text="영상을 준비하고 있습니다...")

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp:
                temp.write(uploaded_video.getbuffer())
                temp_video = temp.name

            info = get_video_info(temp_video)

            progress.progress(10, text="Gemini 서버에 영상을 업로드하고 있습니다...")
            file_info = upload_video_to_gemini(
                temp_video,
                mime_type,
                uploaded_video.name,
            )

            progress.progress(22, text="Gemini가 영상을 처리하고 있습니다...")
            file_info = wait_until_active(file_info)

            progress.progress(38, text="투구 6단계를 분석하고 있습니다...")
            video_result = analyze_video_gemini(file_info)

            coarse_time = float(
                video_result.get("coarse_release_time_seconds", 0.0)
            )
            coarse_time = max(0.0, min(coarse_time, info["duration"]))

            progress.progress(55, text="릴리스 주변 프레임을 추출하고 있습니다...")
            contact_sheet, candidate_times = create_contact_sheet(
                temp_video,
                coarse_time,
                info["duration"],
            )

            progress.progress(70, text="15개 후보 프레임을 비교하고 있습니다...")
            frame_result = refine_release(
                contact_sheet,
                candidate_times,
            )

            selected = int(frame_result.get("selected_frame_number", 8))
            selected = max(1, min(selected, len(candidate_times)))
            refined_time = candidate_times[selected - 1]

            final_image = extract_frame(
                temp_video,
                refined_time,
            )

            progress.progress(85, text="영상과 이미지를 교차검증하고 있습니다...")
            cross_result = cross_validate(
                final_image,
                refined_time,
                video_result,
            )

            report = make_report(
                info,
                video_result,
                frame_result,
                refined_time,
                cross_result,
            )

            progress.progress(100, text="분석 완료")
            time.sleep(0.3)
            progress.empty()

            st.success("분석이 완료되었습니다.")

            m1, m2, m3 = st.columns(3)
            m1.metric("1차 릴리스 추정", f"{coarse_time:.3f} s")
            m2.metric("정밀 릴리스 추정", f"{refined_time:.3f} s")
            m3.metric(
                "최종 판단",
                str(cross_result.get("release_state", "-")),
            )

            st.divider()

            tab1, tab2, tab3 = st.tabs(
                [
                    "🎞 릴리스 후보",
                    "📷 최종 릴리스",
                    "📊 분석 리포트",
                ]
            )

            with tab1:
                st.subheader("릴리스 후보 15 Frames")
                st.image(
                    contact_sheet,
                    caption=(
                        f"영상 기반 1차 추정 {coarse_time:.3f}초를 중심으로 "
                        "앞뒤 구간을 세분화했습니다."
                    ),
                    use_container_width=True,
                )
                st.write(
                    f"Gemini 선택: **Frame {selected}** · "
                    f"신뢰도 **{frame_result.get('confidence', '-')}**"
                )
                st.write(frame_result.get("reason", "-"))

            with tab2:
                st.subheader("Selected Release Frame")
                st.image(
                    final_image,
                    caption=f"{refined_time:.3f} sec",
                    use_container_width=True,
                )
                st.write(
                    f"최종 상태: **{cross_result.get('release_state', '-')}**"
                )

            with tab3:
                st.markdown(report)
                st.download_button(
                    "⬇️ 분석 리포트 저장",
                    data=report,
                    file_name="AI_Pitching_Analyzer_Report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error("분석 중 오류가 발생했습니다.")
            st.code(
                f"{type(exc).__name__}: {exc}",
                language="text",
            )

        finally:
            if temp_video and os.path.exists(temp_video):
                try:
                    os.remove(temp_video)
                except OSError:
                    pass

else:
    st.info(
        "위에서 투구 영상을 업로드하면 분석 버튼이 나타납니다."
    )
