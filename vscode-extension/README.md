# DeepSeek Coder Studio VS Code Extension

This extension connects VS Code to the `DeepSeek Coder Studio` Python backend and exposes commands for:

- Explaining selected code
- Fixing bugs in selected code
- Refactoring selected code
- Generating tests
- Asking a coding question
- Generating code from a prompt
- Opening an assistant panel

## Configuration

Available settings:

- `deepseekCoderStudio.backendUrl`
- `deepseekCoderStudio.apiKey`
- `deepseekCoderStudio.activeModel`
- `deepseekCoderStudio.timeoutMs`
- `deepseekCoderStudio.temperature`
- `deepseekCoderStudio.maxTokens`
- `deepseekCoderStudio.mode`

## Development

```bash
cd vscode-extension
npm install
npm run compile
```

Press `F5` in VS Code to launch an Extension Host.

## Packaging

```bash
cd vscode-extension
npm run package
```
