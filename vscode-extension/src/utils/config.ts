import * as vscode from "vscode";
import { GenerationParameters } from "../types/api";

export interface ExtensionConfig {
  backendUrl: string;
  apiKey: string;
  activeModel: string;
  timeoutMs: number;
  mode: "local" | "remote";
  generation: GenerationParameters;
}

export function getExtensionConfig(): ExtensionConfig {
  // Read all extension settings from the official VS Code configuration registry.
  const configuration = vscode.workspace.getConfiguration("deepseekCoderStudio");

  return {
    backendUrl: configuration.get<string>("backendUrl", "http://127.0.0.1:8000"),
    apiKey: configuration.get<string>("apiKey", ""),
    activeModel: configuration.get<string>("activeModel", "deepseek-coder-v2-lite-instruct"),
    timeoutMs: configuration.get<number>("timeoutMs", 120000),
    mode: configuration.get<"local" | "remote">("mode", "local"),
    generation: {
      temperature: configuration.get<number>("temperature", 0.2),
      max_new_tokens: configuration.get<number>("maxTokens", 512),
      top_p: 0.95,
      do_sample: true,
      response_format: "text"
    }
  };
}
