#!/usr/bin/env bash
set -euo pipefail

SERVICE_PORT="${SERVICE_PORT:-5001}"
SERVICE_URL="${SERVICE_URL:-http://ai-p301:4123/v1/audio/speech}"
AUTH_TOKEN="${AUTH_TOKEN:-local-dev-key}"
MODEL="${MODEL:-CoRal-project/roest-v3-chatterbox-500m}"
INPUT_TEXT="${INPUT_TEXT:-En lokalplan er en detaljeret fysisk plan for et geografisk område. Lokalplanen er bygget op af en redegørelsesdel og en vedtægtsdel. I redegørelsen beskrives planens intentioner og baggrund samt dens forhold til anden planlægning. I vedtægtsdelen fastlægges planområdets afgrænsning og anvendelse samt planens retsvirkninger. Der optages bestemmelser for udnyttelsen af den enkelte ejendom, herunder byggeriets omfang, udformning og udseende, adgangsforhold, friarealer og beplantning m.v. En lokalplan skal efter reglerne i planloven annonceres og fremlægges som forslag i mindst 8 uger. I denne periode har alle mulighed for at fremkomme med indsigelser og ændringsforslag til planen. Hvis planen ikke ændres væsentlig, kan den herefter vedtages endeligt af Byrådet. Ejendomme, der er omfattet af planen, må kun udstykkes, bebygges eller anvendes i overensstemmelse med planens bestemmelser. Den eksisterende lovlige anvendelse af en ejendom kan dog fortsætte som hidtil. Lokalplanen medfører heller ikke i sig selv pligt til at udføre de anlæg med videre, der er indeholdt i planen.}"
# VOICE stays as a fallback/single-voice override; VOICES drives the random pick.
VOICES_STR="${VOICES:-mic nic}"
read -ra VOICES_ARR <<< "${VOICES_STR}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-wav}"
SPEED="${SPEED:-1.0}"
REQUEST_COUNT="${REQUEST_COUNT:-24}"
OUTPUT_DIR="${OUTPUT_DIR:-./tmp/audio-speech-$(date +%Y%m%d-%H%M%S)}"

if [ "${#VOICES_ARR[@]}" -lt 1 ]; then
  echo "VOICES must contain at least one voice name" >&2
  exit 1
fi

if ! [[ "${REQUEST_COUNT}" =~ ^[0-9]+$ ]] || [ "${REQUEST_COUNT}" -lt 1 ] || [ "${REQUEST_COUNT}" -gt 100 ]; then
  echo "REQUEST_COUNT must be an integer between 1 and 100" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

run_request() {
  local index="$1"

  # Pick a voice at random for this request. Each backgrounded call is its
  # own subshell (own PID), and bash >= 5.1 reseeds $RANDOM on fork, so this
  # is safe under `&` — no shared/correlated sequence across parallel jobs.
  local voice="${VOICES_ARR[$((RANDOM % ${#VOICES_ARR[@]}))]}"

  local audio_file="${OUTPUT_DIR}/speech-${index}-${voice}.wav"
  local headers_file="${OUTPUT_DIR}/speech-${index}-${voice}.headers"
  local meta_file="${OUTPUT_DIR}/speech-${index}-${voice}.meta"

  local -a curl_args=(
    -sS
    "${SERVICE_URL}"
    -H "Content-Type: application/json"
    -D "${headers_file}"
    -o "${audio_file}"
    -X POST
    -d "{\"model\":\"${MODEL}\",\"input\":\"${INPUT_TEXT} (${index})\",\"voice\":\"${voice}\",\"response_format\":\"${RESPONSE_FORMAT}\",\"speed\":${SPEED}}"
    -w "voice=${voice}\nhttp_status=%{http_code}\ntime_total=%{time_total}\nsize_download=%{size_download}\n"
  )

  if [ -n "${AUTH_TOKEN}" ]; then
    curl_args+=( -H "Authorization: Bearer ${AUTH_TOKEN}" )
  fi

  curl "${curl_args[@]}" > "${meta_file}"
}

echo "Sending ${REQUEST_COUNT} parallel requests to ${SERVICE_URL}"
echo "Voices in rotation: ${VOICES_ARR[*]}"
echo "Saving outputs under ${OUTPUT_DIR}"

pids=()
for index in $(seq 1 "${REQUEST_COUNT}"); do
  run_request "${index}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

echo ""
echo "Voice distribution:"
for v in "${VOICES_ARR[@]}"; do
  count=$(ls "${OUTPUT_DIR}"/speech-*-"${v}".wav 2>/dev/null | wc -l)
  echo "  ${v}: ${count}"
done

if [ "${failed}" -ne 0 ]; then
  echo "One or more requests failed" >&2
  exit 1
fi

echo "All requests completed"