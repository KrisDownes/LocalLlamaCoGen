import { ChildProcess } from 'child_process';

export interface LLMOptions {
    modelPath?: string;
    apiUrl?: string;
    apiKeyName?: string;
    model?: string;
    systemPrompt?: string;
    replace?: boolean;
    useLocalModel?: boolean;
    maxTokens?: number;
    url?: string;
    completionMode?: boolean;  // Add this for completion-specific behavior
    temperature?: number;      // Add temperature control
}

export interface LocalModelProcess {
    process: ChildProcess;
    kill: () => void;
}

export interface CompletionContext {
    language: string;
    prefix: string;
    linePrefix: string;
    surroundingCode: string;
}

export interface ModelResponse {
    text: string;
    error?: string;
}

// Add this for managing different provider states
export interface ProviderState {
    isProcessing: boolean;
    lastError?: string;
    lastCompletion?: string;
}