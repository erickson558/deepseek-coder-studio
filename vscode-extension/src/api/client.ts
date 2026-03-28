import axios, { AxiosInstance } from "axios";
import { getExtensionConfig } from "../utils/config";
import { ChatRequest, GenerateRequest, InferenceResponse, TaskRequest } from "../types/api";

export class BackendClient {
  private readonly client: AxiosInstance;

  constructor() {
    // Reuse one configured HTTP client per command execution to keep backend calls consistent.
    const config = getExtensionConfig();
    this.client = axios.create({
      baseURL: config.backendUrl,
      timeout: config.timeoutMs,
      headers: config.apiKey ? { "X-API-Key": config.apiKey } : undefined
    });
  }

  async generate(payload: GenerateRequest): Promise<InferenceResponse> {
    // Route one-shot code generation requests to the backend.
    const response = await this.client.post<InferenceResponse>("/generate", payload);
    return response.data;
  }

  async chat(payload: ChatRequest): Promise<InferenceResponse> {
    // Route conversational assistant requests to the backend.
    const response = await this.client.post<InferenceResponse>("/chat", payload);
    return response.data;
  }

  async task(endpoint: string, payload: TaskRequest): Promise<InferenceResponse> {
    // Route task-specific actions such as explain/fix/refactor/tests.
    const response = await this.client.post<InferenceResponse>(endpoint, payload);
    return response.data;
  }
}
