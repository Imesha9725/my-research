function asText(x) {
  if (x == null) return '';
  return String(x);
}

function tokenize(text) {
  // Simple, language-agnostic tokenization (good enough for a baseline ROUGE report).
  // Lowercase + keep letters/numbers; collapse whitespace.
  return asText(text)
    .toLowerCase()
    .replace(/[^a-z0-9\s]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function ngrams(tokens, n) {
  const out = new Map();
  if (!Array.isArray(tokens) || tokens.length < n) return out;
  for (let i = 0; i <= tokens.length - n; i += 1) {
    const g = tokens.slice(i, i + n).join(' ');
    out.set(g, (out.get(g) || 0) + 1);
  }
  return out;
}

function overlapCount(aCounts, bCounts) {
  let n = 0;
  for (const [k, a] of aCounts.entries()) {
    const b = bCounts.get(k) || 0;
    n += Math.min(a, b);
  }
  return n;
}

function prf(overlap, predTotal, refTotal) {
  const precision = predTotal > 0 ? overlap / predTotal : 0;
  const recall = refTotal > 0 ? overlap / refTotal : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return { precision, recall, f1 };
}

function rougeN(candidateTokens, referenceTokens, n) {
  const cand = ngrams(candidateTokens, n);
  const ref = ngrams(referenceTokens, n);
  const overlap = overlapCount(cand, ref);
  const candTotal = [...cand.values()].reduce((s, v) => s + v, 0);
  const refTotal = [...ref.values()].reduce((s, v) => s + v, 0);
  return prf(overlap, candTotal, refTotal);
}

function lcsLength(a, b) {
  // DP LCS length in O(|a|*|b|). For chatbot replies, sequences are short enough.
  const n = a.length;
  const m = b.length;
  if (n === 0 || m === 0) return 0;
  const dp = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i += 1) {
    let prev = 0;
    for (let j = 1; j <= m; j += 1) {
      const tmp = dp[j];
      if (a[i - 1] === b[j - 1]) dp[j] = prev + 1;
      else dp[j] = Math.max(dp[j], dp[j - 1]);
      prev = tmp;
    }
  }
  return dp[m];
}

function rougeL(candidateTokens, referenceTokens) {
  const lcs = lcsLength(candidateTokens, referenceTokens);
  return prf(lcs, candidateTokens.length, referenceTokens.length);
}

function mean(items) {
  if (!items.length) return 0;
  return items.reduce((s, x) => s + x, 0) / items.length;
}

/**
 * Compute ROUGE scores for a dataset.
 *
 * Input examples should be { candidate, reference } OR { prediction, reference }.
 * Returns corpus-level averages and optional per-example scores.
 */
export function computeRougeDataset(examples, { returnPerExample = false } = {}) {
  const rows = Array.isArray(examples) ? examples : [];
  const per = [];
  for (const ex of rows) {
    const candText = ex?.candidate ?? ex?.prediction ?? ex?.pred ?? ex?.chatbot_reply ?? '';
    const refText = ex?.reference ?? ex?.ref ?? ex?.reference_reply ?? '';
    const candToks = tokenize(candText);
    const refToks = tokenize(refText);
    const r1 = rougeN(candToks, refToks, 1);
    const r2 = rougeN(candToks, refToks, 2);
    const rl = rougeL(candToks, refToks);
    per.push({ rouge1: r1, rouge2: r2, rougeL: rl });
  }

  const rouge1_f1 = mean(per.map((x) => x.rouge1.f1));
  const rouge2_f1 = mean(per.map((x) => x.rouge2.f1));
  const rougeL_f1 = mean(per.map((x) => x.rougeL.f1));

  const out = {
    n: per.length,
    rouge1: {
      precision: mean(per.map((x) => x.rouge1.precision)),
      recall: mean(per.map((x) => x.rouge1.recall)),
      f1: rouge1_f1,
    },
    rouge2: {
      precision: mean(per.map((x) => x.rouge2.precision)),
      recall: mean(per.map((x) => x.rouge2.recall)),
      f1: rouge2_f1,
    },
    rougeL: {
      precision: mean(per.map((x) => x.rougeL.precision)),
      recall: mean(per.map((x) => x.rougeL.recall)),
      f1: rougeL_f1,
    },
  };

  if (returnPerExample) out.perExample = per;
  return out;
}

