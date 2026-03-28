import * as vscode from "vscode";

export class AssistantPanel {
  private static currentPanel: AssistantPanel | undefined;
  private readonly panel: vscode.WebviewPanel;

  private constructor(extensionUri: vscode.Uri) {
    this.panel = vscode.window.createWebviewPanel(
      "deepseekCoderStudioPanel",
      "DeepSeek Coder Studio",
      vscode.ViewColumn.Beside,
      {
        enableScripts: false,
        retainContextWhenHidden: true
      }
    );
    this.panel.onDidDispose(() => {
      AssistantPanel.currentPanel = undefined;
    });
    this.panel.webview.html = this.render("Assistant ready.", "Use a command from the Command Palette.");
  }

  static createOrShow(extensionUri: vscode.Uri): AssistantPanel {
    if (AssistantPanel.currentPanel) {
      AssistantPanel.currentPanel.panel.reveal(vscode.ViewColumn.Beside);
      return AssistantPanel.currentPanel;
    }

    AssistantPanel.currentPanel = new AssistantPanel(extensionUri);
    return AssistantPanel.currentPanel;
  }

  update(title: string, requestSummary: string, responseText: string): void {
    this.panel.title = `DeepSeek Coder Studio: ${title}`;
    this.panel.webview.html = this.render(requestSummary, responseText);
  }

  private render(requestSummary: string, responseText: string): string {
    return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <style>
      body { font-family: Segoe UI, sans-serif; padding: 16px; color: #1f2937; }
      h1 { font-size: 18px; margin-bottom: 12px; }
      .card { border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
      pre { white-space: pre-wrap; word-break: break-word; }
    </style>
  </head>
  <body>
    <h1>DeepSeek Coder Studio</h1>
    <div class="card">
      <strong>Request</strong>
      <pre>${escapeHtml(requestSummary)}</pre>
    </div>
    <div class="card">
      <strong>Response</strong>
      <pre>${escapeHtml(responseText)}</pre>
    </div>
  </body>
</html>`;
  }
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
