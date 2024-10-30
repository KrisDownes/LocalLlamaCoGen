import * as vscode from 'vscode';
import * as https from 'https';
import { IncomingMessage, ClientRequest } from 'http';
import * as path from 'path';
import * as child_process from 'child_process';

interface LLMOptions {
    modelPath?: string;
    apiUrl?: string;
    apiKeyName?: string;
    model?: string;
    systemPrompt?: string;
    replace?: boolean;
    useLocalModel?: boolean;
    maxTokens?: number;
    url?: string;
}

interface LocalModelProcess {
    process: child_process.ChildProcess;
    kill: () => void;
}

export class LLMExtension {
    private activeRequest: ClientRequest | null = null;
    private localModelProcess: LocalModelProcess | null = null;
    private outputBuffer: string = '';  // Added for output buffering
    private writeTimeout: NodeJS.Timeout | null = null;

    constructor(private context: vscode.ExtensionContext) {}

    activate() {
        const invokeCommand = vscode.commands.registerCommand('vscode-local-llm.invoke', () => {
            this.invokeLLM();
        });

        const cancelCommand = vscode.commands.registerCommand('vscode-local-llm.cancel', () => {
            this.cancelRequest();
        });

        this.context.subscriptions.push(invokeCommand, cancelCommand);
    }

    private getConfig() {
        const config = vscode.workspace.getConfiguration('localLLM');
        return {
            modelPath: config.get<string>('modelPath', ''),
            apiUrl: config.get<string>('apiUrl', ''),
            apiKeyName: config.get<string>('apiKeyName', ''),
            model: config.get<string>('model', ''),
            systemPrompt: config.get<string>('systemPrompt', 'You are a helpful assistant.'),
            replaceSelection: config.get<boolean>('replaceSelection', false),
            useLocalModel: config.get<boolean>('useLocalModel', false),
            maxTokens: config.get<number>('maxTokens', 4096)
        };
    }

    private async getSelectedText(): Promise<string | undefined> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return undefined;

        const selection = editor.selection;
        if (selection.isEmpty) {
            const cursorPosition = selection.active;
            const range = new vscode.Range(
                new vscode.Position(0, 0),
                cursorPosition
            );
            return editor.document.getText(range);
        }
        return editor.document.getText(selection);
    }

    private async writeTextAtCursor(text: string) {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        await editor.edit((editBuilder) => {
            const position = editor.selection.active;
            editBuilder.insert(position, text);
        });
    }

    private cancelRequest() {
        if (this.activeRequest) {
            this.activeRequest.destroy();
            this.activeRequest = null;
            vscode.window.showInformationMessage('API request cancelled');
        }
        if (this.localModelProcess) {
            this.localModelProcess.kill();
            this.localModelProcess = null;
            vscode.window.showInformationMessage('Local model inference cancelled');
        }
    }

    private invokeLocalModel(prompt: string, opts: LLMOptions): Promise<void> {
        return new Promise((resolve, reject) => {
            if (!opts.modelPath) {
                reject(new Error('Model path not configured'));
                return;
            }
    
            const llamaDir = path.join(__dirname, 'llama');
            const pythonScript = path.join(llamaDir, 'llama_inference.py');
            const args = [
                '--model-path', opts.modelPath,
                '--prompt', prompt,
                '--max-tokens', opts.maxTokens?.toString() || '4096',
                '--system-prompt', opts.systemPrompt || ''
            ];
    
            const env = {
                ...process.env,
                PYTHONPATH: `${llamaDir}:${process.env.PYTHONPATH || ''}`
            };
    
            const childProcess = child_process.spawn('python', [pythonScript, ...args], {
                cwd: llamaDir,
                env: env
            });
    
            this.localModelProcess = {
                process: childProcess,
                kill: () => childProcess.kill()
            };
            
            childProcess.stdout.on('data', async (data: Buffer) => {
                const text = data.toString();
                await this.writeTextAtCursor(text);
            });
    
            childProcess.stderr.on('data', (data: Buffer) => {
                console.error(`Error from local model: ${data.toString()}`);
            });
    
            childProcess.on('close', (code) => {
                this.localModelProcess = null;
                if (code === 0) {
                    resolve();
                } else {
                    reject(new Error(`Local model process exited with code ${code}`));
                }
            });
        });
    }

    private makeAnthropicRequest(opts: LLMOptions, prompt: string): ClientRequest {
        const apiKey = opts.apiKeyName ? process.env[opts.apiKeyName] : undefined;
        
        const data = {
            system: opts.systemPrompt,
            messages: [{ role: 'user', content: prompt }],
            model: opts.model,
            stream: true,
            max_tokens: opts.maxTokens || 4096
        };

        const requestOptions = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
                ...(apiKey && { 'x-api-key': apiKey })
            }
        };

        return this.makeStreamingRequest(opts.url!, data, requestOptions, this.handleAnthropicResponse.bind(this));
    }

    private makeOpenAIRequest(opts: LLMOptions, prompt: string): ClientRequest {
        const apiKey = opts.apiKeyName ? process.env[opts.apiKeyName] : undefined;
        
        const data = {
            messages: [
                { role: 'system', content: opts.systemPrompt },
                { role: 'user', content: prompt }
            ],
            model: opts.model,
            temperature: 0.7,
            stream: true
        };

        const requestOptions = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(apiKey && { 'Authorization': `Bearer ${apiKey}` })
            }
        };

        return this.makeStreamingRequest(opts.url!, data, requestOptions, this.handleOpenAIResponse.bind(this));
    }

    private makeStreamingRequest(
        url: string, 
        data: any, 
        options: https.RequestOptions, 
        handleResponse: (chunk: string) => void
    ): ClientRequest {
        const request = https.request(url, options, (response: IncomingMessage) => {
            response.setEncoding('utf8');
            let buffer = '';

            response.on('data', (chunk: string) => {
                buffer += chunk;
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    handleResponse(line);
                }
            });

            response.on('end', () => {
                if (buffer) {
                    handleResponse(buffer);
                }
                this.activeRequest = null;
            });
        });

        request.on('error', (error) => {
            vscode.window.showErrorMessage(`API request failed: ${error.message}`);
            this.activeRequest = null;
        });

        request.write(JSON.stringify(data));
        request.end();

        return request;
    }

    private handleAnthropicResponse(line: string) {
        if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
                const json = JSON.parse(data);
                if (json.delta?.text) {
                    this.writeTextAtCursor(json.delta.text);
                }
            } catch (e) {
                // Ignore parse errors for non-JSON lines
            }
        }
    }

    private handleOpenAIResponse(line: string) {
        if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
                const json = JSON.parse(data);
                if (json.choices?.[0]?.delta?.content) {
                    this.writeTextAtCursor(json.choices[0].delta.content);
                }
            } catch (e) {
                // Ignore parse errors for non-JSON lines
            }
        }
    }

    private async invokeLLM() {
        const config = this.getConfig();
        const prompt = await this.getSelectedText();
        if (!prompt) {
            vscode.window.showErrorMessage('No text selected or cursor position found');
            return;
        }

        const editor = vscode.window.activeTextEditor;
        if (config.replaceSelection && editor && !editor.selection.isEmpty) {
            await editor.edit((editBuilder) => {
                editBuilder.delete(editor.selection);
            });
        }

        const opts: LLMOptions = {
            url: config.apiUrl,
            apiKeyName: config.apiKeyName,
            model: config.model,
            systemPrompt: config.systemPrompt,
            replace: config.replaceSelection,
            modelPath: config.modelPath,
            useLocalModel: config.useLocalModel,
            maxTokens: config.maxTokens
        };

        try {
            if (config.useLocalModel) {
                if (!config.modelPath) {
                    vscode.window.showErrorMessage('Local model path not configured');
                    return;
                }
                await this.invokeLocalModel(prompt, opts);
            } else {
                if (!config.apiUrl || !config.model) {
                    vscode.window.showErrorMessage('API URL and model must be configured for cloud inference');
                    return;
                }
                if (opts.url?.includes('anthropic')) {
                    this.activeRequest = this.makeAnthropicRequest(opts, prompt);
                } else {
                    this.activeRequest = this.makeOpenAIRequest(opts, prompt);
                }
            }
        } catch (error:any) {
            vscode.window.showErrorMessage(`Error during LLM inference: ${error.message}`);
        }
    }
}

export function activate(context: vscode.ExtensionContext) {
    const extension = new LLMExtension(context);
    extension.activate();
}

export function deactivate() {}

