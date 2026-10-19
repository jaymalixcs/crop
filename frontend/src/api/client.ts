export type Prediction = { condition: string; score: number }
export type Detection = { class_name: string; confidence: number; bbox: number[] }
export type Severity = { ratio: number; level: string }
export type DetectorPrediction = { class_name: string; confidence: number }
export type Recommendation = {
  crop: string
  condition: string
  description: string
  general_management: string[]
  prevention: string[]
  monitoring: string[]
  expert_verification: boolean
}
export type Analysis = {
  id?: number
  crop: string
  condition: string
  is_healthy: boolean
  confidence: number
  model_score: number
  top_predictions: Prediction[]
  recommendations?: Recommendation
  below_threshold?: boolean
  image_url?: string
  created_at?: string
  detections: Detection[]
  severity?: Severity
  detection_severity?: Severity
  detector_prediction?: DetectorPrediction
  annotated_image_base64?: string
}
export type Dashboard = {
  total_analyses: number
  healthy_count: number
  attention_needed_count: number
  most_detected_crop?: string
  most_detected_condition?: string
}
export type Crops = Record<string, string[]>

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}/api${path}`, init)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.detail || 'The analysis service returned an error.')
  return payload as T
}

export function analyzeImage(file: File) {
  const body = new FormData()
  body.append('file', file)
  return request<Analysis>('/analyze', { method: 'POST', body })
}

export const getHistory = () => request<Analysis[]>('/history')
export const getAnalysis = (id: number) => request<Analysis>(`/analysis/${id}`)
export const getDashboard = () => request<Dashboard>('/dashboard')
export const getCrops = () => request<Crops>('/crops')
export const getEvaluation = (forceRerun: boolean = false) =>
  request<any>(`/evaluation${forceRerun ? '?force_rerun=true' : ''}`)
export const runEvaluation = () =>
  request<any>('/evaluation/run', { method: 'POST' })
export const imageUrl = (path?: string) => path ? `${API_BASE_URL}${path}` : undefined

