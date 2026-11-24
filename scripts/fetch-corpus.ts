/**
 * Corpus builder:  npm run corpus
 *
 *  1. shallow-clones (git clone --depth 1) permissively licensed Java repos into data/raw/
 *  2. verifies each licence from its LICENSE file (only Apache-2.0 / MIT are accepted)
 *  3. collects *.java, drops generated/huge/odd files, strips licence headers, normalises
 *     whitespace (CRLF->LF, tabs->4 spaces), de-duplicates (exact + near)
 *  4. splits BY REPOSITORY into train / val / test (whole repos held out)
 *  5. caps each training repo at 15 % of the training bytes (deterministic, hash-ordered)
 *  6. writes data/corpus/{train,val,test}.jsonl, data/corpus/manifest.json, data/SOURCES.md
 *
 * Re-running reuses existing clones (delete data/raw/<repo> to refresh).
 */
import { execFile } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { promisify } from "node:util";
import { cleanJavaFile, detectLicense, nearDupKey, sha1, type DropReason, type LicenseId } from "../src/corpus/clean.js";

const run = promisify(execFile);
const ROOT = resolve(import.meta.dirname, "..");
const RAW = join(ROOT, "data", "raw");
const OUT = join(ROOT, "data", "corpus");

type Split = "train" | "val" | "test";
interface RepoSpec {
  name: string; // owner/repo
  license: LicenseId; // what we expect; verified against the LICENSE file after cloning
  split: Split;
}

/**
 * Splits are decided here, by repository, before any token is counted.
 * Test = 3 whole repos (HTML parser, JSON serialiser, REST client); val = 1 whole repo.
 */
const REPOS: RepoSpec[] = [
  // ---- held-out test repos (never seen by the tokenizer or the model) ----
  { name: "jhy/jsoup", license: "MIT", split: "test" },
  { name: "google/gson", license: "Apache-2.0", split: "test" },
  { name: "square/retrofit", license: "Apache-2.0", split: "test" },
  // ---- validation repo ----
  { name: "apache/commons-codec", license: "Apache-2.0", split: "val" },
  // ---- training repos ----
  { name: "apache/commons-lang", license: "Apache-2.0", split: "train" },
  { name: "apache/commons-collections", license: "Apache-2.0", split: "train" },
  { name: "apache/commons-io", license: "Apache-2.0", split: "train" },
  { name: "apache/commons-text", license: "Apache-2.0", split: "train" },
  { name: "apache/commons-compress", license: "Apache-2.0", split: "train" },
  { name: "apache/commons-csv", license: "Apache-2.0", split: "train" },
  { name: "apache/commons-cli", license: "Apache-2.0", split: "train" },
  { name: "google/guava", license: "Apache-2.0", split: "train" },
  { name: "google/guice", license: "Apache-2.0", split: "train" },
  { name: "square/okhttp", license: "Apache-2.0", split: "train" },
  { name: "square/javapoet", license: "Apache-2.0", split: "train" },
  { name: "mockito/mockito", license: "MIT", split: "train" },
  { name: "FasterXML/jackson-core", license: "Apache-2.0", split: "train" },
  { name: "FasterXML/jackson-databind", license: "Apache-2.0", split: "train" },
  { name: "ReactiveX/RxJava", license: "Apache-2.0", split: "train" },
  { name: "JodaOrg/joda-time", license: "Apache-2.0", split: "train" },
  { name: "TheAlgorithms/Java", license: "MIT", split: "train" },
  { name: "qos-ch/slf4j", license: "MIT", split: "train" },
];

/** Considered because the brief names it, but it is EPL-1.0 (not Apache/MIT): the licence check rejects it. */
const REJECTED_CANDIDATES = ["junit-team/junit4"];

const CAP_FRAC = 0.15;
const ALLOWED: LicenseId[] = ["Apache-2.0", "MIT"];
const DROP_KEYS: DropReason[] = ["too_large", "too_short", "generated", "bad_encoding", "non_ascii", "long_lines", "no_type_decl", "dup_exact", "dup_near"];

const dirOf = (name: string) => join(RAW, name.replace("/", "__"));

async function cloneRepo(name: string): Promise<void> {
  const dir = dirOf(name);
  // A half-finished clone has a .git folder but no resolvable HEAD: wipe it and start again.
  if (existsSync(join(dir, ".git"))) {
    try {
      await run("git", ["-C", dir, "rev-parse", "--verify", "HEAD"]);
      return;
    } catch {
      rmSync(dir, { recursive: true, force: true });
    }
  }
  mkdirSync(RAW, { recursive: true });
  for (let attempt = 1; attempt <= 3; attempt++) {
    console.log(`clone  ${name} (attempt ${attempt})`);
    try {
      // a stalled transfer (seen once on index-pack) is killed after 8 minutes and retried
      await run("git", ["-c", "core.longpaths=true", "clone", "--depth", "1", "--quiet", `https://github.com/${name}.git`, dir], {
        maxBuffer: 1 << 26,
        timeout: 8 * 60 * 1000,
        killSignal: "SIGKILL",
      });
      return;
    } catch (e) {
      rmSync(dir, { recursive: true, force: true });
      if (attempt === 3) throw e;
    }
  }
}

async function commitOf(name: string): Promise<string> {
  const { stdout } = await run("git", ["-C", dirOf(name), "rev-parse", "HEAD"]);
  return stdout.trim();
}

function readLicense(dir: string): LicenseId {
  const cands = readdirSync(dir).filter((f) => /^(LICEN[CS]E|COPYING)/i.test(f));
  for (const f of cands) {
    const id = detectLicense(readFileSync(join(dir, f), "utf8"));
    if (id !== "unknown") return id;
  }
  return "unknown";
}

function* walkJava(dir: string): Generator<string> {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.name === ".git" || e.name === "node_modules") continue;
    const p = join(dir, e.name);
    if (e.isDirectory()) yield* walkJava(p);
    else if (e.isFile() && e.name.endsWith(".java") && e.name !== "module-info.java" && e.name !== "package-info.java") yield p;
  }
}

interface FileRec {
  repo: string;
  path: string;
  text: string;
  key: string; // sha1 of repo:path: deterministic pseudo-random order for the cap
}
interface RepoStats {
  repo: string;
  license: LicenseId;
  commit: string;
  split: Split;
  seen: number;
  kept: number;
  keptBytes: number;
  usedFiles: number;
  usedBytes: number;
  dropped: Partial<Record<DropReason, number>>;
}

async function main() {
  // 1. clone (4 at a time)
  const queue = [...REPOS.map((r) => r.name), ...REJECTED_CANDIDATES];
  await Promise.all(
    Array.from({ length: 4 }, async () => {
      for (let n = queue.shift(); n; n = queue.shift()) {
        try {
          await cloneRepo(n);
        } catch (e) {
          if (REJECTED_CANDIDATES.includes(n)) console.warn(`could not clone ${n}: ${String(e)}`);
          else throw e;
        }
      }
    }),
  );

  // 2. licence check
  const rejected: { repo: string; license: LicenseId; reason: string }[] = [];
  for (const n of REJECTED_CANDIDATES) {
    if (!existsSync(dirOf(n))) continue;
    const lic = readLicense(dirOf(n));
    if (!ALLOWED.includes(lic)) rejected.push({ repo: n, license: lic, reason: `licence ${lic} is not Apache-2.0/MIT` });
  }
  for (const r of REPOS) {
    const lic = readLicense(dirOf(r.name));
    if (!ALLOWED.includes(lic)) throw new Error(`${r.name}: licence ${lic} is not Apache-2.0/MIT, refusing to use it`);
    if (lic !== r.license) throw new Error(`${r.name}: expected ${r.license} but LICENSE file says ${lic}`);
  }

  // 3. collect + clean, train repos first so that val/test lose their duplicates, not train
  const rank = (s: Split) => ["train", "val", "test"].indexOf(s);
  const order = [...REPOS].sort((a, b) => rank(a.split) - rank(b.split));
  const seenExact = new Set<string>();
  const seenNear = new Set<string>();
  const files: FileRec[] = [];
  const stats = new Map<string, RepoStats>();
  for (const r of order) {
    const dir = dirOf(r.name);
    const st: RepoStats = {
      repo: r.name, license: readLicense(dir), commit: await commitOf(r.name), split: r.split,
      seen: 0, kept: 0, keptBytes: 0, usedFiles: 0, usedBytes: 0, dropped: {},
    };
    stats.set(r.name, st);
    const drop = (reason: DropReason) => (st.dropped[reason] = (st.dropped[reason] ?? 0) + 1);
    for (const p of [...walkJava(dir)].sort()) {
      st.seen++;
      const res = cleanJavaFile(readFileSync(p));
      if (!res.ok) { drop(res.reason); continue; }
      if (seenExact.has(sha1(res.text))) { drop("dup_exact"); continue; }
      const nk = nearDupKey(res.text);
      if (seenNear.has(nk)) { drop("dup_near"); continue; }
      seenExact.add(sha1(res.text));
      seenNear.add(nk);
      const rel = relative(dir, p).replace(/\\/g, "/");
      files.push({ repo: r.name, path: rel, text: res.text, key: sha1(r.name + ":" + rel) });
      st.kept++;
      st.keptBytes += Buffer.byteLength(res.text);
    }
    console.log(`clean  ${r.name.padEnd(30)} ${r.split.padEnd(5)} seen ${st.seen}  kept ${st.kept}  ${(st.keptBytes / 1e6).toFixed(2)} MB`);
  }

  // 5. per-repo cap for training repos: iterate to a fixed point (cap = 15 % of the capped total)
  const trainRepos = REPOS.filter((r) => r.split === "train").map((r) => stats.get(r.name)!);
  let cap = Infinity;
  for (let i = 0; i < 100; i++) {
    const total = trainRepos.reduce((s, r) => s + Math.min(r.keptBytes, cap), 0);
    const next = CAP_FRAC * total;
    if (Math.abs(next - cap) < 1) break;
    cap = next;
  }
  console.log(`train repo cap: ${(cap / 1e6).toFixed(2)} MB (${CAP_FRAC * 100}% of capped train bytes)`);

  mkdirSync(OUT, { recursive: true });
  const byRepo = new Map<string, FileRec[]>();
  for (const f of files) {
    const l = byRepo.get(f.repo) ?? [];
    l.push(f);
    byRepo.set(f.repo, l);
  }
  const lines: Record<Split, string[]> = { train: [], val: [], test: [] };
  for (const r of REPOS) {
    const st = stats.get(r.name)!;
    const fs = (byRepo.get(r.name) ?? []).sort((a, b) => (a.key < b.key ? -1 : 1));
    let used = 0;
    for (const f of fs) {
      const bytes = Buffer.byteLength(f.text);
      if (r.split === "train" && used + bytes > cap) continue;
      used += bytes;
      st.usedFiles++;
      lines[r.split].push(JSON.stringify({ repo: f.repo, path: f.path, text: f.text }));
    }
    st.usedBytes = used;
  }
  for (const s of ["train", "val", "test"] as Split[]) writeFileSync(join(OUT, `${s}.jsonl`), lines[s].join("\n") + "\n");
  const all = [...stats.values()];
  const bySplit = (s: Split) => all.filter((r) => r.split === s);
  const manifest = {
    created: new Date().toISOString(),
    cap_frac: CAP_FRAC,
    train_repo_cap_bytes: Math.round(cap),
    rejected,
    repos: all,
    splits: Object.fromEntries(
      (["train", "val", "test"] as Split[]).map((s) => [s, {
        repos: bySplit(s).map((r) => r.repo),
        files: bySplit(s).reduce((a, r) => a + r.usedFiles, 0),
        bytes: bySplit(s).reduce((a, r) => a + r.usedBytes, 0),
      }]),
    ) as Record<Split, { repos: string[]; files: number; bytes: number }>,
  };
  writeFileSync(join(OUT, "manifest.json"), JSON.stringify(manifest, null, 1));

  // 6. SOURCES.md
  const md: string[] = [];
  md.push("# Corpus sources", "");
  md.push("All code is public, permissively licensed (**Apache-2.0** or **MIT**). Each licence was verified by");
  md.push("reading the repository's `LICENSE` file (see `src/corpus/clean.ts: detectLicense`), not assumed.");
  md.push("The raw clones live in `data/raw/` and the cleaned shards in `data/corpus/` (both gitignored); this");
  md.push("file is generated by `npm run corpus` (`scripts/fetch-corpus.ts`).", "");
  md.push("Splits are **by repository** (no file of a held-out repo is ever seen in training; files of a held-out");
  md.push("repo that duplicate a training file were dropped from the held-out side).", "");
  md.push("| Repo | Licence | Commit | Split | Java files seen | kept after cleaning | used |");
  md.push("|---|---|---|---|---:|---:|---:|");
  for (const s of ["train", "val", "test"] as Split[])
    for (const r of bySplit(s))
      md.push(`| [${r.repo}](https://github.com/${r.repo}) | ${r.license} | \`${r.commit.slice(0, 10)}\` | ${r.split} | ${r.seen} | ${r.kept} (${(r.keptBytes / 1e6).toFixed(2)} MB) | ${r.usedFiles} (${(r.usedBytes / 1e6).toFixed(2)} MB) |`);
  md.push("");
  md.push(`Training repos are capped at ${CAP_FRAC * 100} % of the training bytes (cap ${(cap / 1e6).toFixed(2)} MB), selecting files in hash order, so that no`);
  md.push("single code base dominates a 1M-parameter model.", "");
  if (rejected.length) {
    md.push("## Considered and rejected", "");
    for (const r of rejected) md.push(`- \`${r.repo}\`: ${r.reason} (EPL is weak copyleft; this corpus is Apache-2.0/MIT only).`);
    md.push("");
  }
  md.push("## Also considered, not used", "");
  md.push("- `ben-manes/caffeine` (Apache-2.0) and `zxing/zxing` (Apache-2.0): their shallow clones stalled in `git index-pack` on this machine (three attempts, 8-minute timeout); the corpus already reached the token target without them.");
  md.push("");
  md.push("## Cleaning rules", "");
  md.push("- only `*.java` (no `module-info.java` / `package-info.java`); raw file <= 100 KB, cleaned file >= 200 B");
  md.push("- dropped: generated code (`@Generated`, protobuf, \"DO NOT EDIT\"), invalid UTF-8, > 2 % non-ASCII, lines > 600 chars, no type declaration");
  md.push("- licence/copyright header comments stripped; CRLF to LF, tabs to 4 spaces, trailing spaces removed, 4+ blank lines collapsed");
  md.push("- de-duplicated: exact (hash of the cleaned file) and near (hash of the first 2 KB of the body without `package`/`import` lines)", "");
  md.push("## Drop statistics (files)", "");
  md.push("| Repo | " + DROP_KEYS.join(" | ") + " |");
  md.push("|---|" + "---:|".repeat(DROP_KEYS.length));
  for (const r of all) md.push(`| ${r.repo} | ` + DROP_KEYS.map((k) => r.dropped[k] ?? 0).join(" | ") + " |");
  md.push("");
  writeFileSync(join(ROOT, "data", "SOURCES.md"), md.join("\n"));
  const t = manifest.splits;
  console.log(`done: train ${t.train.files} files ${(t.train.bytes / 1e6).toFixed(1)} MB | val ${t.val.files} / ${(t.val.bytes / 1e6).toFixed(1)} MB | test ${t.test.files} / ${(t.test.bytes / 1e6).toFixed(1)} MB`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
