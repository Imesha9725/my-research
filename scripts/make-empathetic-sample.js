#!/usr/bin/env node
/**
 * Create a small sample JSON file from the cleaned EmpatheticDialogues JSONL.
 *
 * Usage:
 *   node scripts/make-empathetic-sample.js --n 200
 *
 * Output:
 *   data/empatheticdialogues/cleaned/sample.json
 */
const fs = require("node:fs");
const path = require("node:path");
const readline = require("node:readline");

const ROOT = path.resolve(__dirname, "..");
const INPUT = path.join(
  ROOT,
  "data",
  "empatheticdialogues",
  "cleaned",
  "empatheticdialogues.all.jsonl",
);
const OUTPUT = path.join(
  ROOT,
  "data",
  "empatheticdialogues",
  "cleaned",
  "sample.json",
);

function parseArgs(argv) {
  const args = { n: 200 };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--n") {
      const v = Number.parseInt(argv[i + 1] ?? "", 10);
      if (Number.isFinite(v) && v > 0) args.n = v;
      i++;
    }
  }
  return args;
}

async function main() {
  const { n } = parseArgs(process.argv);

  if (!fs.existsSync(INPUT)) {
    throw new Error(
      `Missing input: ${path.relative(ROOT, INPUT)} (run prepare-empathetic-dialogues.js first)`,
    );
  }

  const rl = readline.createInterface({
    input: fs.createReadStream(INPUT, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  const sample = [];
  for await (const line of rl) {
    if (!line) continue;
    sample.push(JSON.parse(line));
    if (sample.length >= n) break;
  }
  rl.close();

  fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
  fs.writeFileSync(OUTPUT, JSON.stringify(sample, null, 2), "utf8");

  process.stdout.write(
    `Wrote ${sample.length} conversations -> ${path.relative(ROOT, OUTPUT)}\n`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

