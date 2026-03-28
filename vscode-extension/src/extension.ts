import * as vscode from "vscode";
import { registerCodeActionCommands, registerPromptCommands } from "./commands/codeActions";
import { AssistantPanel } from "./panels/assistantPanel";

export function activate(context: vscode.ExtensionContext): void {
  // Register all commands during extension activation.
  registerCodeActionCommands(context);
  registerPromptCommands(context);

  context.subscriptions.push(
    vscode.commands.registerCommand("deepseekCoderStudio.openAssistantPanel", () => {
      // Create or focus the assistant panel on demand.
      AssistantPanel.createOrShow(context.extensionUri);
    })
  );
}

export function deactivate(): void {
  void vscode.commands.executeCommand("setContext", "deepseekCoderStudio.active", false);
}
