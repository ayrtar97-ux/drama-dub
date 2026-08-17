/**
 * dub-drama.js
 * Input: a full drama episode (3-5 min), already the right length — nothing is trimmed.
 * Output: final_output.mp4
 *   - full episode, multi-character Burmese AI dub (2+ distinct voices, gender-matched)
 *   - original background music/SFX preserved (original dialogue removed via
 *     Demucs vocal separation, new Burmese dub mixed on top of the instrumental bed)
 *   - converted to 9:16 vertical (TikTok/Shorts format) with blurred background padding
 *   - NO subtitles, NO watermark removal by default (see WATERMARK_BOX below)
 *
 * Requirements:
 *   npm install @google/genai
 *   ffmpeg + ffprobe installed
 *   edge-tts installed (pip install edge-tts) — free, no API key needed
 *   demucs installed (pip install demucs) — free, no API key needed, but pulls
 *     in PyTorch so the first install/run takes a few extra minutes in CI. If
 *     it fails for any reason, the pipeline automatically falls back to
 *     dub-only audio (no background bed) rather than failing the whole run.
 *
 * Env vars required:
 *   GEMINI_API_KEY
 *
 * Optional env vars:
 *   EDGE_TTS_RATE          (default: "+10%")
 *   REMOVE_WATERMARK       ("true" to enable delogo — off by default since
 *                            different drama sources have watermarks in
 *                            different places, or none at all)
 *   WATERMARK_X / _Y / _W / _H   (only used if REMOVE_WATERMARK=true)
 *
 * Usage:
 *   node dub-drama.js episode.mp4 final_output.mp4
 */

const { GoogleGenAI } = require("@google/genai");
const { execSync, execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const [, , INPUT_PATH, OUTPUT_PATH = "final_output.mp4"] = process.argv;
if (!INPUT_PATH) {
  console.error("Usage: node dub-drama.js <episode.mp4> [final_output.mp4]");
  process.exit(1);
}

const EDGE_TTS_RATE = process.env.EDGE_TTS_RATE || "+10%";
const REMOVE_WATERMARK = process.env.REMOVE_WATERMARK === "true";
const WATERMARK_BOX = {
  x: Number(process.env.WATERMARK_X || 1000),
  y: Number(process.env.WATERMARK_Y || 580),
  w: Number(process.env.WATERMARK_W || 280),
  h: Number(process.env.WATERMARK_H || 60),
};

// Pool of distinguishable Burmese voice "identities". Only 2 native edge-tts
// Burmese voices exist, so beyond that we reuse them with a pitch shift to
// keep characters at least somewhat distinguishable.
const VOICE_POOL = [
  { voice: "my-MM-ThihaNeural", pitch: "+0Hz" },   // male, normal
  { voice: "my-MM-NilarNeural", pitch: "+0Hz" },   // female, normal
  { voice: "my-MM-ThihaNeural", pitch: "-30Hz" },  // male, lower/older-sounding
  { voice: "my-MM-NilarNeural", pitch: "+30Hz" },  // female, higher/younger-sounding
];

async function withRetry(fn, { retries = 5, baseDelayMs = 5000 } = {}) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const status = err && (err.status || (err.error && err.error.code));
      const isTransient = status === 503 || status === 429;
      if (!isTransient || attempt === retries) throw err;
      const delay = baseDelayMs * Math.pow(2, attempt - 1);
      console.warn(`Gemini call failed (status ${status}), retrying in ${delay / 1000}s...`);
      await new Promise((r) => setTimeout(r, delay));
    }
  }
}

async function main() {
  const tmpDir = fs.mkdtempSync("/tmp/drama-dub-");
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

  // ---- 1. Upload episode to Gemini and get a multi-speaker dialogue script ----
  console.log("Uploading episode to Gemini...");
  const uploaded = await ai.files.upload({ file: INPUT_PATH, config: { mimeType: "video/mp4" } });
  let file = await ai.files.get({ name: uploaded.name });
  while (file.state === "PROCESSING") {
    process.stdout.write(".");
    await new Promise((r) => setTimeout(r, 3000));
    file = await ai.files.get({ name: uploaded.name });
  }
  if (file.state === "FAILED") throw new Error("Gemini file processing failed.");
  console.log("\nGenerating multi-speaker Burmese dialogue script...");

  const prompt = `
This is a short drama episode (3-5 minutes). Watch and listen to it, then write a
Burmese dub script covering the ENTIRE episode from start to end — every line of
dialogue and any important narration/action beats need a corresponding cue.

Rules:
- Break it into cues of 1-8 seconds each (roughly one line of dialogue per cue),
  covering the full episode duration with no large unexplained gaps.
- Cues must be in chronological order and must not overlap.
- Identify each distinct speaking character and label them consistently as
  "speaker": "A", "B", "C", etc. (reuse the same letter for the same character
  throughout the whole episode — double check you haven't split one character
  into two labels or merged two characters into one).
- For "gender", listen to the actual voice pitch/timbre of that speaker as your
  PRIMARY signal (not just visual appearance/clothing/hair, which can mislead).
  Cross-check against how they're addressed by other characters if names/titles
  are used (e.g. "he/she", "Mr./Mrs.", gendered names) to confirm. Answer
  "male" or "female" — pick the closer match even if slightly uncertain, don't
  leave it ambiguous.
- "burmese" must be natural, spoken, conversational Burmese matching the tone/
  emotion of that line (dramatic, tender, angry, etc. as appropriate) — this is
  a full dialogue dub, not a narrated summary, so translate/adapt each line as
  something that character would actually say.
- PACING: write each "burmese" line so it can naturally be SPOKEN OUT LOUD within
  roughly the cue's (end - start) duration at a normal conversational pace.
  Don't write a long sentence for a 1-second reaction shot, and don't write a
  clipped fragment for a 6-second monologue — match the line's length to how
  long that character is actually seen/heard speaking. This matters a lot for
  keeping the dub in sync with the video.
- Timestamps in MM:SS format, relative to this video.

Return ONLY valid JSON, no markdown, no explanation, in this exact shape:
[{"start":"MM:SS","end":"MM:SS","speaker":"A","gender":"male","burmese":"..."}]
`.trim();

  const result = await withRetry(() =>
    ai.models.generateContent({
      model: "gemini-3.5-flash-lite",
      contents: [
        { role: "user", parts: [{ fileData: { fileUri: file.uri, mimeType: file.mimeType } }, { text: prompt }] },
      ],
      config: {
        responseMimeType: "application/json",
        maxOutputTokens: 8192,
        thinkingConfig: { thinkingBudget: 1024 },
      },
    })
  );

  const responseText = result.text;
  if (!responseText) {
    console.error("Gemini returned no text. Full response:");
    console.error(JSON.stringify(result, null, 2));
    throw new Error("Gemini returned empty text.");
  }

  const rawText = responseText.trim().replace(/^```json|```$/g, "").trim();
  let cues;
  try {
    cues = JSON.parse(rawText);
  } catch (e) {
    console.warn("Strict parse failed, attempting recovery...");
    cues = recoverJsonArray(rawText);
    if (!cues || cues.length === 0) throw e;
  }
  console.log(`Got ${cues.length} dialogue cues.`);
  fs.writeFileSync(path.join(tmpDir, "cues.json"), JSON.stringify(cues, null, 2));

  // ---- 2. Assign a consistent voice identity to each unique speaker ----
  const speakerIds = [...new Set(cues.map((c) => c.speaker))];
  const speakerVoiceMap = {};
  let maleIdx = 0, femaleIdx = 0;
  for (const sp of speakerIds) {
    const cueForSpeaker = cues.find((c) => c.speaker === sp);
    const isFemale = (cueForSpeaker.gender || "").toLowerCase() === "female";
    if (isFemale) {
      speakerVoiceMap[sp] = VOICE_POOL.filter((v) => v.voice.includes("Nilar"))[femaleIdx % 2];
      femaleIdx++;
    } else {
      speakerVoiceMap[sp] = VOICE_POOL.filter((v) => v.voice.includes("Thiha"))[maleIdx % 2];
      maleIdx++;
    }
  }
  console.log("Speaker -> voice assignment:", speakerVoiceMap);

  // ---- 3. Generate TTS per cue using each speaker's assigned voice ----
  console.log("Generating multi-character Burmese dub with edge-tts...");
  const audioClips = [];
  for (let i = 0; i < cues.length; i++) {
    const cue = cues[i];
    const startSec = toSeconds(cue.start);
    const endSec = toSeconds(cue.end);
    const slotDuration = Math.max(0.4, endSec - startSec);
    const { voice, pitch } = speakerVoiceMap[cue.speaker];

    const rawMp3 = path.join(tmpDir, `voice_raw_${i}.mp3`);
    edgeTTS(cue.burmese, rawMp3, voice, pitch);

    const rawDuration = getDuration(rawMp3);
    const fittedWav = path.join(tmpDir, `voice_fit_${i}.wav`);
    fitAudioToSlot(rawMp3, rawDuration, slotDuration, fittedWav);

    audioClips.push({ path: fittedWav, startSeconds: startSec });
    process.stdout.write(".");
  }
  console.log("\nDub generation done.");

  // ---- 4. Build the full dialogue audio track ----
  const videoDuration = getDuration(INPUT_PATH);
  const dubTrack = path.join(tmpDir, "dub.wav");
  buildAudioTrack(audioClips, videoDuration, dubTrack);

  // ---- 4b. Isolate original background music/SFX (remove the original speech)
  //          so the dub sits on top of the real soundtrack instead of dead silence.
  let finalAudioTrack = dubTrack;
  try {
    console.log("Separating original vocals from music/SFX with Demucs (this can take a few minutes)...");
    const backgroundBed = separateBackgroundAudio(INPUT_PATH, tmpDir, videoDuration);
    finalAudioTrack = path.join(tmpDir, "dub_with_background.wav");
    mixDubWithBackground(dubTrack, backgroundBed, videoDuration, finalAudioTrack);
    console.log("Background music/SFX preserved and mixed under the new dub.");
  } catch (err) {
    console.warn("Background separation failed, falling back to dub-only audio:", err.message);
    finalAudioTrack = dubTrack;
  }

  // ---- 5. Final ffmpeg pass: (optional watermark removal) + 9:16 vertical + mux new audio ----
  console.log("Rendering final video (9:16 vertical + multi-voice dub)...");
  const { width: vidW, height: vidH } = getDimensions(INPUT_PATH);

  let cleanStage;
  if (REMOVE_WATERMARK) {
    const { x, y, w, h } = clampBoxToFrame(WATERMARK_BOX, vidW, vidH);
    cleanStage = `[0:v]delogo=x=${x}:y=${y}:w=${w}:h=${h}:show=0[clean]`;
  } else {
    cleanStage = `[0:v]null[clean]`;
  }

  const filterComplex = [
    cleanStage,
    `[clean]split=2[bgsrc][fgsrc]`,
    `[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:8[bg]`,
    `[fgsrc]scale=1080:-2[fg]`,
    `[bg][fg]overlay=(W-w)/2:(H-h)/2[v]`,
  ].join(";");

  execSync(
    `ffmpeg -y -i "${INPUT_PATH}" -i "${finalAudioTrack}" ` +
      `-filter_complex "${filterComplex}" ` +
      `-map "[v]" -map 1:a ` +
      `-c:v libx264 -crf 20 -preset veryfast -c:a aac -shortest "${OUTPUT_PATH}"`,
    { stdio: "inherit" }
  );

  console.log(`\nDone. Final multi-voice dubbed video: ${OUTPUT_PATH}`);
}

// ---------- helpers ----------

function toSeconds(mmss) {
  const [m, s] = mmss.split(":").map(Number);
  return m * 60 + s;
}

function getDuration(filePath) {
  return parseFloat(
    execSync(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${filePath}"`)
      .toString()
      .trim()
  );
}

function getDimensions(filePath) {
  const out = execSync(
    `ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "${filePath}"`
  ).toString().trim();
  const [width, height] = out.split("x").map(Number);
  return { width, height };
}

function clampBoxToFrame(box, frameW, frameH) {
  const margin = 2;
  let { x, y, w, h } = box;
  w = Math.min(w, frameW - margin * 2);
  h = Math.min(h, frameH - margin * 2);
  x = Math.min(Math.max(x, 0), frameW - w - margin);
  y = Math.min(Math.max(y, 0), frameH - h - margin);
  return { x, y, w, h };
}

function edgeTTS(text, outPath, voice, pitch) {
  const textFile = outPath + ".txt";
  fs.writeFileSync(textFile, text, "utf8");
  execFileSync(
    "edge-tts",
    ["--voice", voice, "--rate", EDGE_TTS_RATE, "--pitch", pitch, "--file", textFile, "--write-media", outPath],
    { stdio: "ignore" }
  );
}

function fitAudioToSlot(inputPath, rawDuration, slotDuration, outPath) {
  let factor = rawDuration / slotDuration;
  factor = Math.max(1.0, Math.min(3.0, factor)); // only speed up, never slow down; cap at 3x total
  if (factor > 2.5) {
    console.warn(
      `  [timing] Line needed ${factor.toFixed(2)}x speedup to fit its slot — ` +
      `consider shortening that cue's Burmese text for better sync.`
    );
  }
  // atempo only accepts 0.5-2.0 per instance, so chain two filters for anything above 2x.
  const atempoChain =
    factor > 2.0
      ? `atempo=2.0,atempo=${(factor / 2.0).toFixed(3)}`
      : `atempo=${factor.toFixed(3)}`;
  execSync(
    `ffmpeg -y -i "${inputPath}" -filter:a "${atempoChain},volume=0.85" -ar 44100 -ac 2 "${outPath}"`,
    { stdio: "ignore" }
  );
}

// Runs Demucs (free, open-source vocal separation) on the original video's
// audio and returns the path to a WAV containing everything EXCEPT the
// original spoken dialogue (music, ambience, sound effects) — this becomes
// the bed the new Burmese dub sits on top of, instead of dead silence.
function separateBackgroundAudio(videoPath, tmpDir, videoDuration) {
  const originalAudio = path.join(tmpDir, "original_audio.wav");
  execSync(`ffmpeg -y -i "${videoPath}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "${originalAudio}"`, {
    stdio: "ignore",
  });

  const demucsOutDir = path.join(tmpDir, "demucs_out");
  execFileSync(
    "demucs",
    ["--two-stems", "vocals", "-n", "htdemucs", "-o", demucsOutDir, originalAudio],
    { stdio: "inherit" }
  );

  // Demucs writes to <out>/<model_name>/<input_basename>/no_vocals.wav
  const baseName = path.basename(originalAudio, path.extname(originalAudio));
  const noVocalsPath = path.join(demucsOutDir, "htdemucs", baseName, "no_vocals.wav");
  if (!fs.existsSync(noVocalsPath)) {
    throw new Error(`Demucs output not found at expected path: ${noVocalsPath}`);
  }
  return noVocalsPath;
}

// Mixes the new dub (full volume) with the isolated background bed (lowered
// slightly so it doesn't compete with the new voice), trimmed to video length.
function mixDubWithBackground(dubPath, backgroundPath, totalDuration, outPath) {
  execSync(
    `ffmpeg -y -i "${dubPath}" -i "${backgroundPath}" ` +
      `-filter_complex "[1:a]volume=0.55,atrim=0:${totalDuration}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[aout]" ` +
      `-map "[aout]" -t ${totalDuration} "${outPath}"`,
    { stdio: "inherit" }
  );
}

function buildAudioTrack(clips, totalDuration, outPath) {
  const inputs = clips.map((c) => `-i "${c.path}"`).join(" ");
  const delayFilters = clips
    .map((c, i) => `[${i}:a]adelay=${Math.round(c.startSeconds * 1000)}|${Math.round(c.startSeconds * 1000)}[a${i}]`)
    .join(";");
  const mixInputs = clips.map((_, i) => `[a${i}]`).join("");
  const filterComplex = `${delayFilters};${mixInputs}amix=inputs=${clips.length}:duration=longest:dropout_transition=0[aout]`;
  execSync(
    `ffmpeg -y ${inputs} -filter_complex "${filterComplex}" -map "[aout]" -t ${totalDuration} "${outPath}"`,
    { stdio: "inherit" }
  );
}

function recoverJsonArray(rawText) {
  const objects = [];
  let depth = 0, startIdx = -1;
  for (let i = 0; i < rawText.length; i++) {
    const ch = rawText[i];
    if (ch === "{") { if (depth === 0) startIdx = i; depth++; }
    else if (ch === "}") {
      depth--;
      if (depth === 0 && startIdx !== -1) {
        try { objects.push(JSON.parse(rawText.slice(startIdx, i + 1))); } catch (_) {}
        startIdx = -1;
      }
    }
  }
  return objects;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
