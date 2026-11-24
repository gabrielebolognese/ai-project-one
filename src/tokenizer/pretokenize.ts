/**
 * Java-aware pre-tokenisation. BPE merges never cross a pre-token boundary, so this regex decides
 * what a token can ever contain. It partitions ANY string exactly (join(pieces) === input).
 *
 * Alternatives, in priority order:
 *   1. `\n+ *`                      newline(s) plus the following indentation, one piece (so a
 *                                   whole indented line break can become a single token)
 *   2. ` ?identifier`               [letter _ $][letter digit _ $]*, optional single leading space
 *   3. ` ?number`                   0x1F, 10L, 1.5f, 1e10
 *   4. ` ?punctuation-run`          operators/brackets, e.g. `);`  ` ==`  `->`  `<`
 *   5. ` +(?= \S)`                  runs of >= 2 spaces (alignment), leaving 1 for the next word
 *   6. ` +`                         any other run of spaces
 *   7. `[\s\S]`                     catch-all (tabs, stray unicode digits...) so nothing is lost
 *
 * Kept free of node-only APIs so the browser lab can import it.
 */
export const PRETOKEN_SOURCE =
  "\\n+ *|" +
  " ?[\\p{L}_$][\\p{L}\\p{N}_$]*|" +
  " ?\\d[\\p{L}\\p{N}_]*(?:\\.\\d[\\p{L}\\p{N}_]*)?|" +
  " ?[^\\s\\p{L}\\p{N}_$]+|" +
  " +(?= \\S)|" +
  " +|" +
  "[\\s\\S]";
export const PRETOKEN_FLAGS = "gu";

export function makePretokenRegex(source = PRETOKEN_SOURCE, flags = PRETOKEN_FLAGS): RegExp {
  return new RegExp(source, flags);
}

export function pretokenize(text: string, re: RegExp = makePretokenRegex()): string[] {
  re.lastIndex = 0;
  return text.match(re) ?? [];
}
