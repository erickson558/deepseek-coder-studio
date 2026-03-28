import * as vscode from "vscode";

export interface EditorContext {
  selection: string;
  fileContent: string;
  language: string;
  filePath?: string;
}

export function getEditorContext(requireSelection = true): EditorContext | undefined {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    void vscode.window.showWarningMessage("No active editor found.");
    return undefined;
  }

  const selection = editor.document.getText(editor.selection).trim();
  if (requireSelection && !selection) {
    void vscode.window.showWarningMessage("Select some code before running this command.");
    return undefined;
  }

  return {
    selection,
    fileContent: editor.document.getText(),
    language: editor.document.languageId,
    filePath: editor.document.uri.fsPath
  };
}

export async function applyResultToEditor(content: string, mode: "replace" | "insertBelow" | "newDocument"): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor || mode === "newDocument") {
    const document = await vscode.workspace.openTextDocument({
      language: editor?.document.languageId,
      content
    });
    await vscode.window.showTextDocument(document, { preview: false });
    return;
  }

  await editor.edit((editBuilder) => {
    if (mode === "replace") {
      editBuilder.replace(editor.selection, content);
      return;
    }

    const insertPosition = editor.selection.end;
    editBuilder.insert(insertPosition, `\n${content}\n`);
  });
}

export async function copyToClipboard(content: string): Promise<void> {
  await vscode.env.clipboard.writeText(content);
  void vscode.window.showInformationMessage("Assistant result copied to clipboard.");
}
