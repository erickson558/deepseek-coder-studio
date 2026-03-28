import axios, { AxiosInstance } from "axios";
import { getExtensionConfig } from "../utils/config";
import { ChatRequest, GenerateRequest, InferenceResponse, TaskRequest } from "../types/api";

export class BackendClient {
  private readonly client: AxiosInstance;

  constructor() {
    const config = getExtensionConfig();
    this.client = axios.create({
      baseURL: config.backendUrl,
      timeout: config.timeoutMs,
      headers: config.apiKey ? { "X-API-Key": config.apiKey } : undefined
    });
  }

  async generate(payload: GenerateRequest): Promise<InferenceResponse> {
    const response = await this.client.post<InferenceResponse>("/generate", payload);
    return response.data;
  }

  async chat(payload: ChatRequest): Promise<InferenceResponse> {
    const response = await this.client.post<InferenceResponse>("/chat", payload);
    return response.data;
  }

  async task(endpoint: string, payload: TaskRequest): Promise<InferenceResponse> {
    const response = await this.client.post<InferenceResponse>(endpoint, payload);
    return response.data;
  }
}
