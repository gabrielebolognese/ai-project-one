import assert from "node:assert/strict";
import { test } from "node:test";
import { cleanJavaFile, detectLicense, normalizeWhitespace, stripLicenseHeader } from "./clean.js";

const BODY = `package a.b;

import java.util.List;

/** Doc for the class that should stay. */
public class Foo {
    int x = 1;
    List<String> names() { return null; }
    void bar() { System.out.println("hello world from a sufficiently long method body"); }
}
`;

test("strips a block-comment licence header but keeps class javadoc", () => {
  const src = `/*\n * Licensed to the Apache Software Foundation (ASF)\n * under one or more agreements.\n */\n${BODY}`;
  const out = stripLicenseHeader(src);
  assert.ok(out.startsWith("package a.b;"));
  assert.ok(out.includes("Doc for the class that should stay"));
});

test("strips // copyright headers and keeps non-licence leading comments", () => {
  assert.ok(stripLicenseHeader(`// Copyright 2020 Foo Inc.\n// All rights reserved.\n${BODY}`).startsWith("package"));
  const kept = stripLicenseHeader(`/* Helper to do X. */\n${BODY}`);
  assert.ok(kept.startsWith("/* Helper"));
});

test("normalises CRLF, tabs and trailing blanks", () => {
  assert.equal(normalizeWhitespace("a\t\r\n  b   \r\n\r\n\r\n\r\n\r\nc"), "a\n  b\n\n\nc\n");
  assert.equal(normalizeWhitespace("\tx"), "    x\n");
});

test("drops generated, huge, short and non-type files; keeps real ones", () => {
  assert.equal(cleanJavaFile(Buffer.from("@Generated\n" + BODY)).ok, false);
  assert.equal(cleanJavaFile(Buffer.alloc(200 * 1024, "a")).ok, false);
  assert.equal(cleanJavaFile(Buffer.from("class A{}")).ok, false);
  const ok = cleanJavaFile(Buffer.from(BODY.replace(/ {4}/g, "\t").replace(/\n/g, "\r\n")));
  assert.ok(ok.ok && ok.text.includes("    int x = 1;") && !ok.text.includes("\r") && !ok.text.includes("\t"));
});

test("detects licences from LICENSE text", () => {
  assert.equal(detectLicense("Apache License\n Version 2.0, January 2004"), "Apache-2.0");
  assert.equal(detectLicense("The MIT License\n\nCopyright..."), "MIT");
  assert.equal(detectLicense("Eclipse Public License - v 1.0"), "EPL");
  assert.equal(detectLicense("all rights reserved"), "unknown");
});
