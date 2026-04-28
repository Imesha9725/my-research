import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { computeRougeDataset } from './rouge.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function usage() {
  // eslint-disable-next-line no-console
  console.log(
    [
      'Usage:',
      '  node eval/rouge_cli.js <path-to-json>',
      '',
      'JSON format:',
      '  { "examples": [ { "candidate": "...", "reference": "..." }, ... ] }',
      'or simply:',
      '  [ { "candidate": "...", "reference": "..." }, ... ]',
      '',
      'Tip: for your dataset you can also use keys:',
      '  chatbot_reply + reference_reply',
    ].join('\n')
  );
}

const fileArg = process.argv[2];
if (!fileArg) {
  usage();
  process.exit(1);
}

const p = path.isAbsolute(fileArg) ? fileArg : path.resolve(__dirname, '..', fileArg);
if (!fs.existsSync(p)) {
  // eslint-disable-next-line no-console
  console.error('File not found:', p);
  process.exit(1);
}

let data;
try {
  data = JSON.parse(fs.readFileSync(p, 'utf-8'));
} catch (e) {
  // eslint-disable-next-line no-console
  console.error('Invalid JSON:', e?.message || e);
  process.exit(1);
}

const examples = Array.isArray(data) ? data : data?.examples;
const scores = computeRougeDataset(examples, { returnPerExample: false });

// eslint-disable-next-line no-console
console.log(JSON.stringify(scores, null, 2));

