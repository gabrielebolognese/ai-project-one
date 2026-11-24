/**
 * CodeMirror 6 editor with Java highlighting and ghost-text completion.
 *   Tab accepts the ghost text, Esc dismisses it. Without ghost text Tab indents.
 *   Hovering a token chip in the network highlights its characters here.
 */
import { EditorState, StateEffect, StateField, Prec, type Extension } from "@codemirror/state";
import {
  Decoration, EditorView, WidgetType, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection,
  type DecorationSet,
} from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, insertTab } from "@codemirror/commands";
import { HighlightStyle, bracketMatching, indentOnInput, indentUnit, syntaxHighlighting } from "@codemirror/language";
import { java } from "@codemirror/lang-java";
import { tags as t } from "@lezer/highlight";
import { C, FONT_MONO } from "./theme";

interface Ghost {
  pos: number;
  text: string;
}

const setGhost = StateEffect.define<Ghost | null>();
const setHighlight = StateEffect.define<{ from: number; to: number } | null>();

class GhostWidget extends WidgetType {
  constructor(readonly text: string) { super(); }
  eq(o: GhostWidget): boolean { return o.text === this.text; }
  toDOM(): HTMLElement {
    const s = document.createElement("span");
    s.className = "cm-ghost";
    s.textContent = this.text;
    const k = document.createElement("span");
    k.className = "cm-ghost-key";
    k.textContent = "Tab";
    s.appendChild(k);
    return s;
  }
  ignoreEvent(): boolean { return true; }
}

const ghostField = StateField.define<Ghost | null>({
  create: () => null,
  update(g, tr) {
    for (const e of tr.effects) if (e.is(setGhost)) return e.value;
    if (tr.docChanged || tr.selection) return null;
    return g;
  },
  provide: (f) => EditorView.decorations.from(f, (g): DecorationSet =>
    g && g.text ? Decoration.set([Decoration.widget({ widget: new GhostWidget(g.text), side: 1 }).range(g.pos)]) : Decoration.none),
});

const hlMark = Decoration.mark({ class: "cm-token-hl" });
const highlightField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(d, tr) {
    for (const e of tr.effects) if (e.is(setHighlight)) {
      const v = e.value;
      if (!v || v.to <= v.from) return Decoration.none;
      const to = Math.min(v.to, tr.state.doc.length);
      const from = Math.max(0, Math.min(v.from, to));
      return to > from ? Decoration.set([hlMark.range(from, to)]) : Decoration.none;
    }
    return tr.docChanged ? d.map(tr.changes) : d;
  },
  provide: (f) => EditorView.decorations.from(f),
});

const labHighlight = HighlightStyle.define([
  { tag: [t.keyword, t.modifier, t.controlKeyword, t.definitionKeyword], color: C.blue },
  { tag: [t.typeName, t.className, t.namespace], color: C.teal },
  { tag: [t.string, t.character], color: C.yellow },
  { tag: [t.number, t.bool, t.null], color: C.orange },
  { tag: [t.comment, t.lineComment, t.blockComment], color: "#6b7686", fontStyle: "italic" },
  { tag: [t.function(t.variableName), t.function(t.propertyName)], color: "#e8c7ff" },
  { tag: [t.annotation, t.meta], color: C.pink },
  { tag: [t.operator, t.punctuation, t.bracket], color: "#aab4c3" },
  { tag: [t.variableName, t.propertyName], color: C.text },
]);

const labTheme = EditorView.theme(
  {
    "&": { color: C.text, backgroundColor: "transparent", height: "100%", fontSize: "13.5px" },
    ".cm-scroller": { fontFamily: FONT_MONO, lineHeight: "1.62" },
    ".cm-content": { caretColor: C.yellow, padding: "10px 0" },
    ".cm-cursor, .cm-dropCursor": { borderLeftColor: C.yellow, borderLeftWidth: "2px" },
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": { backgroundColor: "rgba(88,196,221,0.22)" },
    ".cm-gutters": { backgroundColor: C.panel, color: "#465161", border: "none" },
    ".cm-activeLineGutter": { backgroundColor: "transparent", color: C.dim },
    ".cm-activeLine": { backgroundColor: "rgba(255,255,255,0.025)" },
    ".cm-matchingBracket": { backgroundColor: "rgba(247,217,111,0.18)", outline: "none" },
  },
  { dark: true },
);

export interface LabEditor {
  view: EditorView;
  getCode(): string;
  getCursor(): number;
  showGhost(pos: number, text: string): void;
  clearGhost(): void;
  ghost(): Ghost | null;
  highlight(from: number, to: number): void;
  clearHighlight(): void;
  focus(): void;
}

export function createEditor(parent: HTMLElement, doc: string, onChange: (code: string, cursor: number, docChanged: boolean) => void, onAccept: (text: string) => void): LabEditor {
  const acceptKeys = Prec.highest(
    keymap.of([
      {
        key: "Tab",
        run: (v) => {
          const g = v.state.field(ghostField);
          if (g && g.text && v.state.selection.main.head === g.pos) {
            v.dispatch({ changes: { from: g.pos, insert: g.text }, selection: { anchor: g.pos + g.text.length }, effects: setGhost.of(null), userEvent: "input.complete" });
            onAccept(g.text);
            return true;
          }
          return insertTab(v);
        },
      },
      {
        key: "Escape",
        run: (v) => {
          if (!v.state.field(ghostField)) return false;
          v.dispatch({ effects: setGhost.of(null) });
          return true;
        },
      },
    ]),
  );
  const ext: Extension[] = [
    lineNumbers(), highlightActiveLineGutter(), highlightActiveLine(), drawSelection(), history(), indentOnInput(), bracketMatching(),
    java(), syntaxHighlighting(labHighlight), labTheme, EditorView.lineWrapping, ghostField, highlightField, acceptKeys,
    EditorState.tabSize.of(4),
    indentUnit.of("    "),
    keymap.of([...defaultKeymap, ...historyKeymap]),
    EditorView.updateListener.of((u) => {
      if (u.docChanged || u.selectionSet) onChange(u.state.doc.toString(), u.state.selection.main.head, u.docChanged);
    }),
  ];
  const view = new EditorView({ state: EditorState.create({ doc, extensions: ext, selection: { anchor: doc.length } }), parent });
  return {
    view,
    getCode: () => view.state.doc.toString(),
    getCursor: () => view.state.selection.main.head,
    showGhost: (pos, text) => view.dispatch({ effects: setGhost.of({ pos, text }) }),
    clearGhost: () => { if (view.state.field(ghostField)) view.dispatch({ effects: setGhost.of(null) }); },
    ghost: () => view.state.field(ghostField),
    highlight: (from, to) => view.dispatch({ effects: setHighlight.of({ from, to }) }),
    clearHighlight: () => view.dispatch({ effects: setHighlight.of(null) }),
    focus: () => view.focus(),
  };
}
