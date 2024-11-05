import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as path from 'path';
import { LLMOptions, LocalModelProcess } from './types';

export class LocalLLMCompletionProvider implements vscode.CompletionItemProvider {
    private localModelProcess: LocalModelProcess | null = null;
    private LLAMA_BASE_DIR = path.join(__dirname, '..', 'src', 'llama');

    constructor(private context: vscode.ExtensionContext) {}

    private getConfig(): LLMOptions {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        const config = workspaceFolder 
            ? vscode.workspace.getConfiguration('localLLM', workspaceFolder.uri)
            : vscode.workspace.getConfiguration('localLLM');

        return {
            modelPath: config.get<string>('modelPath') || '',
            systemPrompt: config.get<string>('systemPrompt') || 
                'You are an expert programming assistant. Analyze the code context and provide relevant code completions. Only respond with the actual code completion, no explanations.',
            useLocalModel: true,
            maxTokens: config.get<number>('maxTokens') || 100 // Shorter for completions
        };
    }

    private async getDocumentContext(document: vscode.TextDocument, position: vscode.Position): Promise<string> {
        const linePrefix = document.lineAt(position).text.substr(0, position.character);
        const startLine = Math.max(0, position.line - 10);
        const contextRange = new vscode.Range(
            new vscode.Position(startLine, 0),
            position
        );
        const context = document.getText(contextRange);
        const languageId = document.languageId;

        return `Language: ${languageId}\n\nContext:\n${context}\n\nCurrent line: ${linePrefix}\nSuggest completion for: `;
    }

    private invokeLocalModel(prompt: string, opts: LLMOptions): Promise<string> {
        return new Promise((resolve, reject) => {
            if (!opts.modelPath) {
                reject(new Error('Model path not configured'));
                return;
            }

            const pythonScript = path.join(this.LLAMA_BASE_DIR, 'testing.py');
            const args = [
                '--model-path', opts.modelPath,
                '--prompt', prompt,
                '--max-tokens', opts.maxTokens?.toString() || '100',
                '--system-prompt', opts.systemPrompt || '',
                '--completion-mode', 'true' // Add this flag to your Python script
            ];

            const env = {
                ...process.env,
                PYTHONPATH: `${this.LLAMA_BASE_DIR}:${process.env.PYTHONPATH || ''}`
            };

            let output = '';
            const childProcess = child_process.spawn('python', [pythonScript, ...args], {
                cwd: this.LLAMA_BASE_DIR,
                env: env
            });

            this.localModelProcess = {
                process: childProcess,
                kill: () => childProcess.kill()
            };

            childProcess.stdout.on('data', (data: Buffer) => {
                output += data.toString();
            });

            childProcess.stderr.on('data', (data: Buffer) => {
                console.error(`Error from local model: ${data.toString()}`);
            });

            childProcess.on('close', (code) => {
                this.localModelProcess = null;
                if (code === 0) {
                    resolve(output.trim());
                } else {
                    reject(new Error(`Local model process exited with code ${code}`));
                }
            });
        });
    }

    async provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): Promise<vscode.CompletionItem[]> {
        try {
            const config = this.getConfig();
            if (!config.modelPath) {
                console.error('Model path not configured');
                return [];
            }

            const context = await this.getDocumentContext(document, position);
            const completion = await this.invokeLocalModel(context, config);

            if (!completion || token.isCancellationRequested) {
                return [];
            }

            const completionItem = new vscode.CompletionItem(completion);
            completionItem.kind = vscode.CompletionItemKind.Text;
            completionItem.detail = 'Local LLM Suggestion';
            completionItem.insertText = completion;
            completionItem.command = {
                command: 'editor.action.triggerSuggest',
                title: 'Re-trigger completions'
            };

            return [completionItem];
        } catch (error) {
            console.error('Error providing completion:', error);
            return [];
        }
    }

    dispose() {
        if (this.localModelProcess) {
            this.localModelProcess.kill();
            this.localModelProcess = null;
        }
    }
}