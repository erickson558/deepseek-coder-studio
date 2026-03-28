import * as vscode from "vscode";
import { BackendClient } from "../api/client";
import { AssistantPanel } from "../panels/assistantPanel";
import { GenerateRequest, TaskRequest } from "../types/api";
import { getExtensionConfig } from "../utils/config";
import { applyResultToEditor, copyToClipboard, getEditorContext } from "../utils/editor";

type TaskEndpoint = "/explain" | "/fix" | "/refactor" | "/tests";

export function registerCodeActionCommands(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("deepseekCoderStudio.explainSelectedCode", () => runTask(context, "/explain", "Explain Selected Code")),
    vscode.commands.registerCommand("deepseekCoderStudio.fixSelectedCode", () => runTask(context, "/fix", "Fix Selected Code")),
    vscode.commands.registerCommand("deepseekCoderStudio.refactorSelectedCode", () => runTask(context, "/refactor", "Refactor Selected Code")),
    vscode.commands.registerCommand("deepseekCoderStudio.generateTests", () => runTask(context, "/tests", "Generate Tests"))
  );
}

async function runTask(context: vscode.ExtensionContext, endpoint: TaskEndpoint, title: string): Promise<void> {
  const editorContext = getEditorContext(true);
  if (!editorContext) {
    return;
  }

  const config = getExtensionConfig();
  const client = new BackendClient();
  const panel = AssistantPanel.createOrShow(context.extensionUri);

  const payload: TaskRequest = {
    selection: editorContext.selection,
    file_content: editorContext.fileContent,
    file_path: editorContext.filePath,
    language: editorContext.language,
    model: config.activeModel,
    parameters: config.generation
  };

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title, cancellable: false },
    async () => {
      try {
        const response = await client.task(endpoint, payload);
        panel.update(title, `${endpoint}\n${editorContext.filePath ?? editorContext.language}`, response.output_text);
        await presentResultOptions(response.output_text);
      } catch (error) {
        showError(error);
      }
    }
  );
}

export function registerPromptCommands(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("deepseekCoderStudio.generateCodeFromPrompt", () => generateFromPrompt(context)),
    vscode.commands.registerCommand("deepseekCoderStudio.askAssistant", () => askAssistant(context))
  );
}

async function generateFromPrompt(context: vscode.ExtensionContext): Promise<void> {
  const prompt = await vscode.window.showInputBox({
    title: "Generate Code",
    prompt: "Describe the code you want to generate."
  });
  if (!prompt) {
    return;
  }

  const editorContext = getEditorContext(false);
  const config = getExtensionConfig();
  const client = new BackendClient();
  const panel = AssistantPanel.createOrShow(context.extensionUri);
  const payload: GenerateRequest = {
    prompt,
    context: editorContext?.selection || editorContext?.fileContent,
    language: editorContext?.language,
    model: config.activeModel,
    parameters: config.generation
  };

  try {
    const response = await client.generate(payload);
    panel.update("Generate Code", prompt, response.output_text);
    await presentResultOptions(response.output_text);
  } catch (error) {
    showError(error);
  }
}

async function askAssistant(context: vscode.ExtensionContext): Promise<void> {
  const question = await vscode.window.showInputBox({
    title: "Ask Coding Assistant",
    prompt: "Ask a coding question or request."
  });
  if (!question) {
    return;
  }

  const editorContext = getEditorContext(false);
  const config = getExtensionConfig();
  const client = new BackendClient();
  const panel = AssistantPanel.createOrShow(context.extensionUri);

  const contextText = editorContext?.selection || editorContext?.fileContent;
  const message = contextText
    ? `${question}\n\nCurrent language: ${editorContext?.language}\n\nContext:\n${contextText}`
    : question;

  try {
    const response = await client.chat({
      model: config.activeModel,
      parameters: config.generation,
      messages: [{ role: "user", content: message }]
    });
    panel.update("Ask Assistant", question, response.output_text);
    await presentResultOptions(response.output_text);
  } catch (error) {
    showError(error);
  }
}

async function presentResultOptions(content: string): Promise<void> {
  const choice = await vscode.window.showQuickPick(
    [
      { label: "Replace selection", mode: "replace" as const },
      { label: "Insert below", mode: "insertBelow" as const },
      { label: "Open in new document", mode: "newDocument" as const },
      { label: "Copy to clipboard", mode: "copy" as const }
    ],
    { title: "Apply assistant result" }
  );

  if (!choice) {
    return;
  }
  if (choice.mode === "copy") {
    await copyToClipboard(content);
    return;
  }
  await applyResultToEditor(content, choice.mode);
}

function showError(error: unknown): void {
  const message = error instanceof Error ? error.message : "Unknown backend error";
  void vscode.window.showErrorMessage(`DeepSeek Coder Studio request failed: ${message}`);
}
