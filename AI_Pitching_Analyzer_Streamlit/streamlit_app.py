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
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="AI Pitching Coach", page_icon="⚾", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1550px; padding-top: 1.8rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.18); padding: 12px 14px; border-radius: 14px;}
      .coach-card {border: 1px solid rgba(128,128,128,0.22); border-radius: 16px; padding: 16px 18px; margin-bottom: 14px; background: rgba(128,128,128,0.05);}
      .coach-title {font-size: 1.06rem; font-weight: 700; margin-bottom: 10px;}
      .coach-label {font-size: 0.82rem; font-weight: 700; opacity: 0.72; margin-top: 8px;}
      .cue-box {border-left: 4px solid #f2b134; padding: 9px 12px; margin-top: 8px; background: rgba(242,177,52,0.08); border-radius: 6px; font-weight: 650;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚾ AI Pitching Coach")
st.caption("투구 영상을 단계별로 분석하고, 고정된 코칭 기준에 따라 문제점 · 교정 방법 · 구체적인 연습 Cue를 제공합니다.")

BASE_URL = "https://generativelanguage.googleapis.com"


def get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


API_KEY = get_secret("GEMINI_API_KEY")
SECRET_MODEL = get_secret("MODEL", None)

if not API_KEY:
    st.error("GEMINI_API_KEY가 설정되어 있지 않습니다. Streamlit Community Cloud의 App settings → Secrets에 등록해주세요.")
    st.stop()

MODEL_CANDIDATES = ["gemini-3.5-flash-lite", SECRET_MODEL, "gemini-3.1-flash-lite"]
MODEL_CANDIDATES = list(dict.fromkeys(x for x in MODEL_CANDIDATES if x))

COACHING_RUBRIC = """
다음 6개 항목만 사용하여 평가한다.

1. Balance
- Max leg lift까지 머리와 몸통의 축이 지나치게 좌우로 흔들리지 않는지
- 한 발 지지 상태에서 중심을 비교적 안정적으로 유지하는지
- 정확한 무게중심 수치를 추정하지 말 것

2. Stride
- 앞발이 투구 진행 방향으로 비교적 일관되게 이동하는지
- 보폭이 영상상 지나치게 짧거나 길어 보이는지
- 신장/캘리브레이션 정보가 없으므로 보폭의 정확한 비율을 숫자로 추정하지 말 것

3. Arm timing
- Foot contact 시점에 투구 팔이 지나치게 뒤처져 있지는 않은지
- 하체 착지와 팔 준비 동작의 타이밍이 크게 어긋나 보이는지
- 정확한 어깨/팔꿈치 각도를 측정했다고 주장하지 말 것

4. Hip-trunk sequence
- 하체/골반의 회전과 상체 회전이 완전히 동시에 시작되는지, 또는 하체가 선행하고 상체가 뒤따르는 순서가 관찰되는지
- 2D 영상만으로 회전 각도나 회전 속도를 정량 측정하지 말 것

5. Lead-leg support
- 앞발 착지 후 앞다리가 몸을 지지하는지
- 릴리스로 진행하면서 앞무릎이 계속 크게 무너지는지, 또는 비교적 안정적인 지지 기반을 형성하는지
- 정확한 무릎 각도를 임의로 추정하지 말 것

6. Follow-through
- 릴리스 이후 투구 팔과 몸통의 움직임이 자연스럽게 이어지는지
- 체중 이동이 앞쪽으로 계속되는지
- 동작이 갑자기 끊기거나 균형이 크게 무너지는지

판정은 반드시 다음 3개 중 하나만 사용한다.
- Stable
- Needs attention
- Unable to assess

중요 원칙:
- 영상에서 실제로 명확하게 관찰되는 특징만 근거로 사용한다.
- 문제가 명확하지 않으면 억지로 문제를 만들지 않는다.
- Stable인 항목에 불필요한 교정을 제안하지 않는다.
- 카메라 각도, 프레임, 영상 품질 때문에 판단하기 어렵다면 Unable to assess를 사용한다.
- 통증, 부상 위험, 질환, 의학적 상태를 진단하지 않는다.
- '이 자세가 부상을 유발한다'와 같이 의학적 인과관계를 단정하지 않는다.
""".strip()


def api_error(response):
    try:
        data = response.json()
        return data.get("error", {}).get("message", response.text)
    except Exception:
        return response.text


def call_gemini(parts):
    last_error = None
    for model in MODEL_CANDIDATES:
        url = f"{BASE_URL}/v1beta/models/{model}:generateContent"
        payload = {"contents": [{"role": "user", "parts": parts}]}
        for attempt in range(2):
            try:
                response = requests.post(
                    url,
                    headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"},
                    json=payload,
                    timeout=(20, 180),
                )
            except requests.exceptions.Timeout:
                last_error = f"{model}: 응답 시간 초과"
                if attempt == 0:
                    time.sleep(2)
                    continue
                break

            if response.ok:
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    last_error = f"{model}: 응답 후보가 없습니다."
                    break
                output_parts = candidates[0].get("content", {}).get("parts", [])
                texts = [part["text"] for part in output_parts if "text" in part]
                if texts:
                    return "\n".join(texts), model
                last_error = f"{model}: text 응답이 없습니다."
                break

            status = response.status_code
            message = api_error(response)
            last_error = f"{model} / HTTP {status}: {message}"
            if status in {429, 500, 502, 503, 504}:
                if attempt == 0:
                    time.sleep(3)
                    continue
                break
            if status in {403, 404}:
                break
            raise RuntimeError(last_error)

    raise RuntimeError("Gemini 요청을 완료하지 못했습니다.\n" + (last_error or "알 수 없는 오류"))


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


def upload_video_to_gemini(video_path, mime_type, display_name):
    file_size = os.path.getsize(video_path)
    start_response = requests.post(
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
        raise RuntimeError(f"영상 업로드 준비 실패 {start_response.status_code}: {api_error(start_response)}")
    upload_url = start_response.headers.get("X-Goog-Upload-URL") or start_response.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Gemini upload URL을 받지 못했습니다.")

    with open(video_path, "rb") as f:
        upload_response = requests.post(
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
        raise RuntimeError(f"영상 업로드 실패 {upload_response.status_code}: {api_error(upload_response)}")
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
        response = requests.get(
            f"{BASE_URL}/v1beta/{file_info['name']}",
            headers={"x-goog-api-key": API_KEY},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"영상 상태 확인 실패 {response.status_code}: {api_error(response)}")
        file_info = response.json()
        time.sleep(2)
    raise TimeoutError("Gemini 영상 처리 시간이 5분을 초과했습니다.")


STAGE_PROMPT = """
첨부된 영상은 야구 투구 영상이다.
목표는 '자세 교정'을 바로 수행하는 것이 아니라, 먼저 투구의 핵심 단계를 영상에서 찾아내고 각 단계에서 실제로 관찰되는 사실만 기록하는 것이다.

다음 6개 stage_id를 반드시 사용한다.
1. wind_up
2. max_leg_lift
3. foot_contact
4. arm_cocking
5. release
6. follow_through

각 단계마다:
- key_time_seconds: 해당 동작을 가장 잘 나타내는 영상 시작 기준 초
- observation: 영상에서 직접 확인 가능한 사실
- uncertainty: 판단하기 어려운 부분

특히 release는 공이 손을 떠나는 것으로 보이는 순간을 사용한다.
정확히 보이지 않으면 가장 가능성이 높은 시점을 쓰되 uncertainty에 명시한다.

중요:
- 아직 자세를 좋다/나쁘다 평가하지 말 것
- 정확한 관절 각도를 임의로 추정하지 말 것
- 선수의 통증/부상/의학적 상태를 추정하지 말 것
- 영상에서 안 보이는 부분은 안 보인다고 명시할 것

반드시 JSON 하나만 출력한다.

{
  "summary": "전체 투구 동작의 관찰 요약",
  "stages": [
    {"stage_id":"wind_up","stage_name":"Wind-up","key_time_seconds":0.0,"observation":"","uncertainty":""},
    {"stage_id":"max_leg_lift","stage_name":"Max leg lift","key_time_seconds":0.0,"observation":"","uncertainty":""},
    {"stage_id":"foot_contact","stage_name":"Foot contact","key_time_seconds":0.0,"observation":"","uncertainty":""},
    {"stage_id":"arm_cocking","stage_name":"Arm cocking","key_time_seconds":0.0,"observation":"","uncertainty":""},
    {"stage_id":"release","stage_name":"Release","key_time_seconds":0.0,"observation":"","uncertainty":""},
    {"stage_id":"follow_through","stage_name":"Follow-through","key_time_seconds":0.0,"observation":"","uncertainty":""}
  ],
  "limitations": ["이 영상에서 단계 탐지 시의 한계"]
}
""".strip()


def analyze_pitch_stages(file_info):
    parts = [
        {"file_data": {"mime_type": file_info.get("mimeType", "video/mp4"), "file_uri": file_info["uri"]}},
        {"text": STAGE_PROMPT},
    ]
    text, used_model = call_gemini(parts)
    return parse_json(text), used_model


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
    return {"fps": float(fps), "frames": int(frames), "duration": float(duration), "width": width, "height": height}


def extract_frame(video_path, seconds):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(seconds) * 1000)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError(f"{seconds:.3f}초 프레임을 추출하지 못했습니다.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def stage_time(stage_result, stage_id, default_time, duration):
    for stage in stage_result.get("stages", []):
        if stage.get("stage_id") == stage_id:
            try:
                value = float(stage.get("key_time_seconds"))
                return max(0.0, min(value, duration))
            except Exception:
                pass
    return max(0.0, min(default_time, duration))


def get_key_frames(video_path, stage_result, duration):
    release = stage_time(stage_result, "release", duration * 0.65, duration)
    frame_specs = [
        ("max_leg_lift", "Max leg lift", stage_time(stage_result, "max_leg_lift", release - 1.0, duration)),
        ("foot_contact", "Foot contact", stage_time(stage_result, "foot_contact", release - 0.45, duration)),
        ("release", "Release", release),
        ("follow_through", "Follow-through", stage_time(stage_result, "follow_through", release + 0.45, duration)),
    ]
    results = []
    for stage_id, label, t in frame_specs:
        results.append({"stage_id": stage_id, "label": label, "time": t, "image": extract_frame(video_path, t)})
    return results


def make_keyframe_sheet(key_frames):
    tile_w, tile_h, cols = 440, 330, 2
    sheet = Image.new("RGB", (tile_w * 2, tile_h * 2), "white")
    for i, item in enumerate(key_frames):
        frame = item["image"].copy()
        frame.thumbnail((tile_w - 16, tile_h - 58))
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        x = (tile_w - frame.width) // 2
        tile.paste(frame, (x, 52))
        draw = ImageDraw.Draw(tile)
        draw.text((12, 12), f"{item['label']} | {item['time']:.3f} sec", fill="black")
        sheet.paste(tile, ((i % cols) * tile_w, (i // cols) * tile_h))
    return sheet


def image_to_part(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/jpeg", "data": encoded}}


def evaluate_coaching(stage_result, keyframe_sheet):
    stage_json = json.dumps(stage_result, ensure_ascii=False, indent=2)
    prompt = f"""
당신은 야구 투구 동작을 교육 목적으로 설명하는 AI 코치다.
입력으로 다음 두 가지가 제공된다.
1. 전체 영상에서 추출한 단계별 관찰 결과
2. Max leg lift / Foot contact / Release / Follow-through 핵심 프레임

[단계별 관찰 결과]
{stage_json}

[평가 기준]
{COACHING_RUBRIC}

각 6개 평가 항목에 대하여 반드시 다음 내용을 작성한다.
- status: Stable / Needs attention / Unable to assess
- observation: 영상/프레임에서 직접 관찰되는 사실
- reason: 왜 이 요소를 연습에서 확인할 가치가 있는지
- correction: Needs attention일 경우 구체적인 교정 방법
- cue: 실제 연습 중 기억할 수 있는 짧은 한 문장 지시
- next_check: 다음 촬영에서 무엇을 비교해야 하는지
- confidence: high / medium / low
- evidence_stage: 어떤 단계/프레임을 근거로 했는지

중요:
- 문제가 명확하지 않으면 절대로 Needs attention을 억지로 만들지 않는다.
- Stable이면 correction에는 '현재 패턴 유지'라고 작성한다.
- Unable to assess이면 correction과 cue는 '추가 촬영 필요' 수준으로 작성한다.
- 정확한 각도/거리/속도를 측정했다고 주장하지 않는다.
- 부상, 통증, 질병을 진단하지 않는다.
- 한 영상만으로 이상적인 투구라고 단정하지 않는다.

top_priorities는 실제로 Needs attention으로 판정된 항목만 포함한다.
최대 3개만 선택한다. Needs attention이 하나도 없다면 빈 배열 []로 둔다.

반드시 JSON 하나만 출력한다.

{{
  "overall_coaching_summary": "전체 코칭 요약",
  "evaluations": [
    {{"criterion_id":"balance","criterion_name":"Balance","status":"Stable","observation":"","reason":"","correction":"","cue":"","next_check":"","confidence":"medium","evidence_stage":"Max leg lift"}},
    {{"criterion_id":"stride","criterion_name":"Stride","status":"Stable","observation":"","reason":"","correction":"","cue":"","next_check":"","confidence":"medium","evidence_stage":"Foot contact"}},
    {{"criterion_id":"arm_timing","criterion_name":"Arm timing","status":"Stable","observation":"","reason":"","correction":"","cue":"","next_check":"","confidence":"medium","evidence_stage":"Foot contact"}},
    {{"criterion_id":"hip_trunk_sequence","criterion_name":"Hip-trunk sequence","status":"Stable","observation":"","reason":"","correction":"","cue":"","next_check":"","confidence":"medium","evidence_stage":"Foot contact → Release"}},
    {{"criterion_id":"lead_leg_support","criterion_name":"Lead-leg support","status":"Stable","observation":"","reason":"","correction":"","cue":"","next_check":"","confidence":"medium","evidence_stage":"Foot contact → Release"}},
    {{"criterion_id":"follow_through","criterion_name":"Follow-through","status":"Stable","observation":"","reason":"","correction":"","cue":"","next_check":"","confidence":"medium","evidence_stage":"Follow-through"}}
  ],
  "top_priorities": ["stride", "arm_timing"],
  "limitations": ["코칭 해석의 한계"]
}}
""".strip()
    text, used_model = call_gemini([image_to_part(keyframe_sheet), {"text": prompt}])
    return parse_json(text), used_model


STATUS_META = {
    "Stable": {"emoji": "🟢", "label": "Stable"},
    "Needs attention": {"emoji": "🟡", "label": "Needs attention"},
    "Unable to assess": {"emoji": "⚪", "label": "Unable to assess"},
}


def status_text(status):
    meta = STATUS_META.get(status, {"emoji": "⚪", "label": status or "Unknown"})
    return f"{meta['emoji']} {meta['label']}"


def evaluations_by_id(coaching_result):
    return {item.get("criterion_id"): item for item in coaching_result.get("evaluations", [])}


def coaching_dataframe(coaching_result):
    return pd.DataFrame([
        {
            "평가 기준": item.get("criterion_name", "-"),
            "판정": status_text(item.get("status", "")),
            "근거 단계": item.get("evidence_stage", "-"),
            "신뢰도": item.get("confidence", "-"),
        }
        for item in coaching_result.get("evaluations", [])
    ])


def render_coaching_card(item):
    title = f"{status_text(item.get('status', 'Unable to assess'))} · {item.get('criterion_name', '-')}"
    html = f"""
    <div class="coach-card">
      <div class="coach-title">{title}</div>
      <div class="coach-label">관찰</div><div>{item.get('observation', '-')}</div>
      <div class="coach-label">왜 확인해야 하나요?</div><div>{item.get('reason', '-')}</div>
      <div class="coach-label">교정 방법</div><div>{item.get('correction', '-')}</div>
      <div class="cue-box">💡 {item.get('cue', '-')}</div>
      <div class="coach-label">다음 촬영에서 확인</div><div>{item.get('next_check', '-')}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def make_report(video_info, stage_result, coaching_result, key_frames, models_used):
    model_text = ", ".join(dict.fromkeys(models_used))
    report = f"# ⚾ AI Pitching Coach Report\n\n**실제 사용 Gemini 모델:** `{model_text}`\n\n"
    report += f"## 1. 영상 정보\n- 길이: **{video_info['duration']:.2f}초**\n- FPS: **{video_info['fps']:.2f}**\n- 해상도: **{video_info['width']} × {video_info['height']}**\n\n"
    report += f"## 2. 전체 동작 관찰\n{stage_result.get('summary', '-')}\n\n## 3. 핵심 프레임\n"
    for frame in key_frames:
        report += f"- {frame['label']}: **{frame['time']:.3f}초**\n"
    report += f"\n## 4. 전체 코칭 요약\n{coaching_result.get('overall_coaching_summary', '-')}\n\n## 5. 단계별 코칭\n"
    for item in coaching_result.get("evaluations", []):
        report += f"\n### {status_text(item.get('status'))} {item.get('criterion_name', '-')}\n"
        report += f"**관찰**  \n{item.get('observation', '-')}\n\n"
        report += f"**왜 확인해야 하나요?**  \n{item.get('reason', '-')}\n\n"
        report += f"**교정 방법**  \n{item.get('correction', '-')}\n\n"
        report += f"**Coaching Cue**  \n{item.get('cue', '-')}\n\n"
        report += f"**다음 촬영에서 확인**  \n{item.get('next_check', '-')}\n\n"
    report += "\n---\n본 결과는 교육 목적의 2D 영상 기반 피드백이며, 전문적인 3D 생체역학 측정이나 의학적 평가를 대체하지 않습니다.\n"
    return report


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None

left, right = st.columns([0.95, 1.45], gap="large")

with left:
    st.subheader("📹 My Pitch")
    uploaded_video = st.file_uploader(
        "투구 영상을 업로드하세요",
        type=["mp4", "mov", "avi", "mpeg", "mpg", "webm", "wmv"],
        label_visibility="collapsed",
    )
    if uploaded_video is not None:
        if st.session_state.last_filename != uploaded_video.name:
            st.session_state.analysis_result = None
            st.session_state.last_filename = uploaded_video.name
        st.video(uploaded_video)
        st.caption(f"{uploaded_video.name} · {uploaded_video.size / 1024 / 1024:.1f} MB")
        analyze_clicked = st.button("⚾ 자세 분석 시작", type="primary", use_container_width=True)
    else:
        analyze_clicked = False
        st.info("투구 한 번이 잘 보이는 짧은 영상을 업로드해주세요.")

if analyze_clicked:
    temp_video = None
    try:
        with right:
            progress = st.progress(0, text="영상을 준비하고 있습니다...")

        suffix = Path(uploaded_video.name).suffix or ".mp4"
        mime_type = uploaded_video.type or mimetypes.guess_type(uploaded_video.name)[0] or "video/mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded_video.getbuffer())
            temp_video = temp.name

        video_info = get_video_info(temp_video)

        with right:
            progress.progress(10, text="Gemini 서버에 영상을 업로드하고 있습니다...")
        file_info = upload_video_to_gemini(temp_video, mime_type, uploaded_video.name)

        with right:
            progress.progress(22, text="Gemini가 영상을 처리하고 있습니다...")
        file_info = wait_until_active(file_info)

        with right:
            progress.progress(38, text="투구의 핵심 단계를 찾고 있습니다...")
        stage_result, stage_model = analyze_pitch_stages(file_info)

        with right:
            progress.progress(58, text="핵심 자세 프레임을 추출하고 있습니다...")
        key_frames = get_key_frames(temp_video, stage_result, video_info["duration"])
        keyframe_sheet = make_keyframe_sheet(key_frames)

        with right:
            progress.progress(74, text="6개 코칭 기준으로 자세를 평가하고 있습니다...")
        coaching_result, coaching_model = evaluate_coaching(stage_result, keyframe_sheet)

        models_used = [stage_model, coaching_model]
        report = make_report(video_info, stage_result, coaching_result, key_frames, models_used)

        st.session_state.analysis_result = {
            "video_info": video_info,
            "stage_result": stage_result,
            "coaching_result": coaching_result,
            "key_frames": key_frames,
            "models_used": models_used,
            "report": report,
        }

        with right:
            progress.progress(100, text="코칭 분석 완료")
            time.sleep(0.25)
            progress.empty()

    except Exception as exc:
        with right:
            st.error("분석 중 오류가 발생했습니다.")
            st.code(f"{type(exc).__name__}: {exc}", language="text")
    finally:
        if temp_video and os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except OSError:
                pass

with left:
    result = st.session_state.analysis_result
    if result:
        st.divider()
        st.subheader("📸 Key Frames")
        key_frames = result["key_frames"]
        c1, c2 = st.columns(2)
        for i, item in enumerate(key_frames):
            target = c1 if i % 2 == 0 else c2
            with target:
                st.image(item["image"], caption=f"{item['label']} · {item['time']:.3f}s", use_container_width=True)
        with st.expander("단계 탐지 원문 보기"):
            st.json(result["stage_result"])

with right:
    result = st.session_state.analysis_result
    if not result:
        st.subheader("🎯 Today's Coaching")
        st.info("왼쪽에서 영상을 업로드하고 **자세 분석 시작**을 누르면 이 영역에 맞춤형 코칭 결과가 표시됩니다.")
        st.markdown(
            """
**평가 기준**
- Balance
- Stride
- Arm timing
- Hip-trunk sequence
- Lead-leg support
- Follow-through

각 항목은 `Stable / Needs attention / Unable to assess` 중 하나로 판정합니다.
"""
        )
    else:
        coaching = result["coaching_result"]
        evaluations = evaluations_by_id(coaching)

        st.subheader("🎯 Today's Coaching")
        st.write(coaching.get("overall_coaching_summary", "-"))

        priorities = coaching.get("top_priorities", [])
        if priorities:
            st.markdown("### 우선 교정 포인트")
            for criterion_id in priorities[:3]:
                item = evaluations.get(criterion_id)
                if item:
                    render_coaching_card(item)
        else:
            st.success("현재 영상에서는 명확하게 Needs attention으로 판정된 교정 포인트가 없습니다. 아래 단계별 분석에서 유지할 점과 판단이 어려운 항목을 확인하세요.")

        st.divider()
        st.markdown("### 📊 단계별 코칭 기준")
        df = coaching_dataframe(coaching)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### 세부 피드백")
        for item in coaching.get("evaluations", []):
            with st.expander(f"{status_text(item.get('status'))} {item.get('criterion_name', '-')}"):
                st.write("**관찰**")
                st.write(item.get("observation", "-"))
                st.write("**왜 확인해야 하나요?**")
                st.write(item.get("reason", "-"))
                st.write("**교정 방법**")
                st.write(item.get("correction", "-"))
                st.info("💡 Coaching Cue: " + str(item.get("cue", "-")))
                st.write("**다음 촬영에서 확인**")
                st.write(item.get("next_check", "-"))
                st.caption(f"근거 단계: {item.get('evidence_stage', '-')} · 신뢰도: {item.get('confidence', '-')}")

        st.divider()
        st.caption("실제 사용 모델: " + " → ".join(dict.fromkeys(result["models_used"])))
        st.download_button(
            "⬇️ 전체 코칭 리포트 저장",
            data=result["report"],
            file_name="AI_Pitching_Coach_Report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        with st.expander("⚠️ AI 코칭의 한계"):
            limitations = result["stage_result"].get("limitations", []) + coaching.get("limitations", [])
            if limitations:
                for item in limitations:
                    st.write("• " + str(item))
            else:
                st.write("명시된 추가 한계가 없습니다.")
            st.write("이 결과는 교육 목적의 2D 영상 기반 피드백이며, 전문적인 3D 생체역학 측정이나 의학적 평가를 대체하지 않습니다.")

