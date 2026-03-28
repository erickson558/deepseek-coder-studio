import * as esbuild from "esbuild";
import { rmSync } from "node:fs";

const watch = process.argv.includes("--watch");
rmSync("out", { recursive: true, force: true });

const context = await esbuild.context({
  entryPoints: ["src/extension.ts"],
  bundle: true,
  format: "cjs",
  platform: "node",
  target: "node18",
  outfile: "out/extension.js",
  sourcemap: true,
  external: ["vscode"]
});

if (watch) {
  await context.watch();
  console.log("esbuild watching...");
} else {
  await context.rebuild();
  await context.dispose();
}
