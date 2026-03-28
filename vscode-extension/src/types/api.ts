export interface GenerationParameters {
  temperature: number;
  max_new_tokens: number;
  top_p: number;
  do_sample: boolean;
  response_format: "text" | "json";
}

export interface TaskRequest {
  prompt?: string;
  selection?: string;
  language?: string;
  file_path?: string;
  file_content?: string;
  task_context?: string;
  model?: string;
  parameters: GenerationParameters;
}

export interface GenerateRequest {
  prompt: string;
  context?: string;
  language?: string;
  model?: string;
  parameters: GenerationParameters;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  model?: string;
  parameters: GenerationParameters;
}

export interface InferenceResponse {
  task: string;
  model_id: string;
  output_text: string;
  latency_ms: number;
  output_json?: Record<string, unknown> | null;
}
