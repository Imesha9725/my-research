#!/usr/bin/env node
/**
 * Prepare / clean EmpatheticDialogues dataset into JSONL.
 *
 * Input (expected):
 *   data/empatheticdialogues/empatheticdialogues/{train,valid,test}.csv
 *
 * Output:
 *   data/empatheticdialogues/cleaned/{train,valid,test}.jsonl
 *   data/empatheticdialogues/cleaned/empatheticdialogues.all.jsonl
 *
 * Each JSONL line is a single conversation:
 * {
 *   "split": "train" | "valid" | "test",
 *   "conv_id": "hit:0_conv:1",
 *   "emotion": "sentimental",
 *   "prompt": "...",
 *   "turns": [{"idx":1,"speaker_idx":1,"text":"..."}, ...]
 * }
 */
const fs = require("node:fs");
const path = require("node:path");
const readline = require("node:readline");
const { once } = require("node:events");
const os = require("node:os");
const { createHash } = require("node:crypto");

const ROOT = path.resolve(__dirname, "..");
const INPUT_DIR = path.join(
  ROOT,
  "data",
  "empatheticdialogues",
  "empatheticdialogues",
);
const OUTPUT_DIR = path.join(ROOT, "data", "empatheticdialogues", "cleaned");

const SPLITS = ["train", "valid", "test"];

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function cleanText(s) {
  if (s == null) return "";
  let out = String(s);
  // Dataset uses literal token to stand for commas.
  out = out.replaceAll("_comma_", ",");
  // Normalize whitespace.
  out = out.replace(/\s+/g, " ").trim();
  // Strip trivial empty quotes.
  if (out === '""') out = "";
  return out;
}

function splitCsvLine(line) {
  // Minimal CSV parser that supports quotes and commas.
  // EmpatheticDialogues CSV is simple enough for this (no embedded newlines).
  const fields = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        const next = line[i + 1];
        if (next === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += ch;
      }
    } else {
      if (ch === ",") {
        fields.push(cur);
        cur = "";
      } else if (ch === '"') {
        inQuotes = true;
      } else {
        cur += ch;
      }
    }
  }
  fields.push(cur);
  return fields;
}

function stableIdFromConv(conv) {
  // Useful for downstream: stable short id even if you merge splits.
  const h = createHash("sha1")
    .update(`${conv.conv_id}\n${conv.emotion}\n${conv.prompt}`)
    .digest("hex")
    .slice(0, 12);
  return `ed_${h}`;
}

async function readSplit(split) {
  const inputPath = path.join(INPUT_DIR, `${split}.csv`);
  if (!fs.existsSync(inputPath)) {
    throw new Error(`Missing input file: ${inputPath}`);
  }

  const rl = readline.createInterface({
    input: fs.createReadStream(inputPath, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  const convs = new Map(); // conv_id -> conversation

  let isHeader = true;
  let lineNum = 0;
  for await (const line of rl) {
    lineNum++;
    if (isHeader) {
      isHeader = false;
      continue;
    }
    if (!line) continue;

    const cols = splitCsvLine(line);
    // header: conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags
    if (cols.length < 6) continue;

    const conv_id = cols[0];
    const utterance_idx = Number.parseInt(cols[1], 10);
    const emotion = cleanText(cols[2]);
    const prompt = cleanText(cols[3]);
    const speaker_idx = Number.parseInt(cols[4], 10);
    const utterance = cleanText(cols[5]);
    const tags = cleanText(cols[7] ?? "");

    if (!convs.has(conv_id)) {
      convs.set(conv_id, {
        split,
        conv_id,
        id: null,
        emotion,
        prompt,
        turns: [],
        tags: [],
        source: {
          file: path.relative(ROOT, inputPath).replaceAll("\\", "/"),
          line_count: 0,
        },
      });
    }

    const conv = convs.get(conv_id);
    conv.source.line_count++;
    // Keep first seen emotion/prompt if later rows differ for any reason.
    if (!conv.emotion && emotion) conv.emotion = emotion;
    if (!conv.prompt && prompt) conv.prompt = prompt;
    if (tags) conv.tags.push(tags);

    if (Number.isFinite(utterance_idx) && utterance) {
      conv.turns.push({
        idx: utterance_idx,
        speaker_idx: Number.isFinite(speaker_idx) ? speaker_idx : null,
        text: utterance,
      });
    }

    // Light sanity guard: if something is wildly wrong, stop early.
    if (lineNum === 2 && conv_id === "conv_id") {
      throw new Error("CSV parsing failed: header line was treated as data.");
    }
  }

  // Finalize: sort turns, de-dupe tags, attach stable id.
  for (const conv of convs.values()) {
    conv.turns.sort((a, b) => a.idx - b.idx);
    conv.tags = Array.from(new Set(conv.tags)).filter(Boolean);
    conv.id = stableIdFromConv(conv);
  }

  return Array.from(convs.values());
}

function writeJsonl(filePath, records) {
  const out = fs.createWriteStream(filePath, { encoding: "utf8" });
  for (const r of records) {
    out.write(JSON.stringify(r) + os.EOL);
  }
  out.end();
}

async function main() {
  ensureDir(OUTPUT_DIR);

  const all = [];
  for (const split of SPLITS) {
    const convs = await readSplit(split);
    const outPath = path.join(OUTPUT_DIR, `${split}.jsonl`);
    writeJsonl(outPath, convs);
    all.push(...convs);
    process.stdout.write(
      `Wrote ${convs.length} conversations -> ${path.relative(ROOT, outPath)}\n`,
    );
  }

  const allPath = path.join(OUTPUT_DIR, "empatheticdialogues.all.jsonl");
  writeJsonl(allPath, all);
  process.stdout.write(
    `Wrote ${all.length} conversations -> ${path.relative(ROOT, allPath)}\n`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

