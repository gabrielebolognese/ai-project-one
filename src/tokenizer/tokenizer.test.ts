import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { BOF, EOF, MERGE_BASE, Tokenizer, countPretokens, pretokenize, trainBpe } from "./index.js";

const SAMPLE = `package org.example.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A tiny utility class.
 */
public final class Strings {
    private static final int MAX_LEN = 0x7FFF;
    private static final double RATIO = 1.5e-3;

    private Strings() {
    }

    public static <T extends Comparable<T>> List<T> sorted(final List<T> input) {
        final List<T> out = new ArrayList<>(input);
        out.sort((a, b) -> a.compareTo(b));
        for (int i = 0; i < out.size(); i++) {
            if (out.get(i) == null || i >>> 1 != 0) {
                throw new IllegalArgumentException("null at " + i + "\\n");
            }
        }
        return out;
    }

    @Override
    public String toString() {
        Map<String, Integer> counts = new java.util.HashMap<>();
        counts.merge("héllo wörld", 1, Integer::sum);   // unicode + aligned comment
        long big = 1_000_000L;
        return counts.toString();
    }
}
`;

function trainOnSample(vocab: number): Tokenizer {
  const counts = countPretokens([SAMPLE, SAMPLE.replace(/Strings/g, "Lists"), SAMPLE.replace(/sorted/g, "ordered")]);
  return new Tokenizer(trainBpe(counts, vocab));
}

test("pre-tokenizer partitions any string exactly", () => {
  const cases = [SAMPLE, "", " ", "\n\n\n   x", "a\tb\r\nc", "x   =  y;", "² ٣ \u{1F600} é", "  \n  ", "a  \n"];
  for (const s of cases) assert.equal(pretokenize(s).join(""), s);
});

test("pre-tokenizer keeps indentation, identifiers and operators apart", () => {
  const p = pretokenize("    final int x = a >>> 2;\n        return x;");
  assert.deepEqual(p, ["   ", " final", " int", " x", " =", " a", " >>>", " 2", ";", "\n        ", "return", " x", ";"]);
});

test("pre-token boundaries are never crossed by merges", () => {
  const t = trainOnSample(400);
  for (const { piece, ids } of t.encodePieces(SAMPLE)) assert.equal(t.decode(ids), piece);
});

test("BPE training is deterministic and ids are laid out as documented", () => {
  const a = trainOnSample(350);
  const b = trainOnSample(350);
  assert.deepEqual(a.data.merges, b.data.merges);
  assert.equal(a.data.vocabSize, 350);
  assert.equal(a.data.merges.length, 350 - MERGE_BASE);
  assert.equal(a.tokens[BOF], "<|bof|>");
  assert.equal(a.tokens[EOF], "<|eof|>");
});

test("round trip is exact on Java, including indentation, tabs, CRLF and unicode", () => {
  const t = trainOnSample(512);
  const inputs = [
    SAMPLE,
    SAMPLE.replace(/    /g, "\t"),
    SAMPLE.replace(/\n/g, "\r\n"),
    "",
    "    ",
    "\n\n\n",
    "int x=1;//c\n",
    "😀 αβγ 日本語 \u0000 \u007f",
    SAMPLE.slice(0, 137),
  ];
  for (const s of inputs) {
    const ids = t.encode(s);
    assert.equal(t.decode(ids), s);
    assert.equal(t.decode(t.encode(s, { bof: true, eof: true })), s, "specials decode to nothing");
  }
});

test("encode is shorter than bytes and ids stay in range", () => {
  const t = trainOnSample(512);
  const ids = t.encode(SAMPLE);
  assert.ok(ids.length < Buffer.byteLength(SAMPLE) / 2);
  for (const id of ids) assert.ok(id >= 3 && id < t.vocabSize);
});

test("whole indented line breaks can become single tokens", () => {
  const t = trainOnSample(512);
  const ids = t.encode("\n        ");
  assert.equal(ids.length, 1);
  assert.equal(t.tokens[ids[0]], "\\n        ");
});

test("JSON serialisation round-trips the tokenizer", () => {
  const t = trainOnSample(400);
  const u = Tokenizer.fromJSON(t.toJSON());
  assert.deepEqual(u.encode(SAMPLE), t.encode(SAMPLE));
});

test("released models/tokenizer.json (if built) round-trips Java files of the corpus", { skip: !existsSync(join(process.cwd(), "models", "tokenizer.json")) }, () => {
  const t = Tokenizer.fromJSON(readFileSync(join(process.cwd(), "models", "tokenizer.json"), "utf8"));
  assert.ok(t.vocabSize >= 1024);
  assert.equal(t.decode(t.encode(SAMPLE)), SAMPLE);
  const val = join(process.cwd(), "data", "corpus", "val.jsonl");
  if (existsSync(val)) {
    const rows = readFileSync(val, "utf8").split("\n").filter(Boolean).slice(0, 40);
    for (const l of rows) {
      const { text } = JSON.parse(l) as { text: string };
      assert.equal(t.decode(t.encode(text)), text);
    }
  }
});
