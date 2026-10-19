import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import { Activity, ArrowRight, Award, BarChart3, CheckCircle2, Clock3, FileImage, History as HistoryIcon, Leaf, Menu, ScanLine, ShieldCheck, Sprout, Upload, X, Zap } from 'lucide-react'
import { analyzeImage, getAnalysis, getCrops, getDashboard, getHistory, imageUrl } from './api/client'
import type { Analysis, Crops, Dashboard } from './api/client'
import { EvaluationView } from './components/EvaluationView'
import './App.css'

type View = 'analyze' | 'history' | 'dashboard' | 'crops' | 'evaluation'

function App() {
  const [view, setView] = useState<View>('analyze')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState<Analysis | null>(null)
  const [history, setHistory] = useState<Analysis[]>([])
  const [dashboard, setDashboard] = useState<Dashboard | null>(null)
  const [crops, setCrops] = useState<Crops | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setError('')
    if (view === 'history') getHistory().then(setHistory).catch(showError)
    if (view === 'dashboard') getDashboard().then(setDashboard).catch(showError)
    if (view === 'crops') getCrops().then(setCrops).catch(showError)
  }, [view])

  const showError = (reason: unknown) => setError(reason instanceof Error ? reason.message : 'Could not reach the analysis service.')
  const chooseFile = (next?: File) => {
    if (!next) return
    if (!next.type.startsWith('image/')) { setError('Please choose a JPG, PNG, or WEBP image.'); return }
    setFile(next); setPreview(URL.createObjectURL(next)); setResult(null); setError('')
  }
  const analyze = async () => {
    if (!file) return
    setLoading(true); setError('')
    try { setResult(await analyzeImage(file)) } catch (reason) { showError(reason) } finally { setLoading(false) }
  }
  const reset = () => { if (preview) URL.revokeObjectURL(preview); setFile(null); setPreview(''); setResult(null) }
  const nav = (next: View) => { setView(next); setMenuOpen(false); setError('') }
  const selectHistory = async (item: Analysis) => {
    if (!item.id) return setResult(item)
    try { setResult(await getAnalysis(item.id)); setView('analyze') } catch (reason) { showError(reason) }
  }

  return <div className="app-shell">
    <header className="topbar"><button className="brand" onClick={() => nav('analyze')}><span className="brand-mark"><Sprout size={19} /></span><span>Fieldwise <small>AI CROP HEALTH</small></span></button><button className="menu-button" onClick={() => setMenuOpen(!menuOpen)} aria-label="Toggle navigation"><Menu /></button><nav className={menuOpen ? 'open' : ''}>{([['analyze', 'Analyze'], ['history', 'History'], ['dashboard', 'Dashboard'], ['crops', 'Crops'], ['evaluation', 'Evaluation & Testing']] as [View, string][]).map(([key, label]) => <button key={key} className={view === key ? 'active' : ''} onClick={() => nav(key)}>{key === 'analyze' ? <ScanLine size={16} /> : key === 'history' ? <HistoryIcon size={16} /> : key === 'dashboard' ? <BarChart3 size={16} /> : key === 'crops' ? <Leaf size={16} /> : <Award size={16} />}{label}</button>)}</nav><span className="status"><span /> Model online</span></header>
    <main>{view === 'analyze' && <AnalyzeView file={file} preview={preview} result={result} loading={loading} error={error} inputRef={inputRef} onDrop={(event) => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]) }} onInput={(event) => chooseFile(event.target.files?.[0])} analyze={analyze} clear={reset} />}{view === 'history' && <HistoryView items={history} error={error} onSelect={selectHistory} />}{view === 'dashboard' && <DashboardView data={dashboard} error={error} />}{view === 'crops' && <CropsView data={crops} error={error} />}{view === 'evaluation' && <EvaluationView />}</main>

    <footer><span>Fieldwise AI</span><span>Plant health intelligence for better decisions</span><span><ShieldCheck size={14} /> Research prototype · Verify with a local agronomist</span></footer>
  </div>
}

function AnalyzeView({ file, preview, result, loading, error, inputRef, onDrop, onInput, analyze, clear }: { file: File | null; preview: string; result: Analysis | null; loading: boolean; error: string; inputRef: React.RefObject<HTMLInputElement | null>; onDrop: (event: DragEvent<HTMLDivElement>) => void; onInput: (event: ChangeEvent<HTMLInputElement>) => void; analyze: () => void; clear: () => void }) {
  return <><section className="intro"><div><p className="eyebrow"><Activity size={15} /> FIELD NOTES / LIVE ANALYSIS</p><h1>Know what your crop<br /><em>is telling you.</em></h1><p className="lede">Turn a leaf image into a clear, evidence-led health signal in seconds.</p></div><div className="intro-stat"><span>15</span><p>conditions recognized<br />across 3 crops</p></div></section><section className="workspace"><div className="upload-column"><div className="section-heading"><div><p className="eyebrow">01 / ADD A SAMPLE</p><h2>Upload a leaf image</h2></div><span className="quiet">JPG · PNG · WEBP</span></div><div className={preview ? 'dropzone has-image' : 'dropzone'} onDrop={onDrop} onDragOver={(event) => event.preventDefault()}>{preview ? <><img src={preview} alt="Selected crop leaf" /><button className="remove" onClick={clear} aria-label="Remove image"><X size={16} /></button><div className="image-label"><FileImage size={15} /> {file?.name}</div></> : <><div className="upload-icon"><Upload size={22} /></div><strong>Drop your image here</strong><span>or browse from your device</span><button className="outline-button" onClick={() => inputRef.current?.click()}>Choose image <ArrowRight size={15} /></button><input ref={inputRef} type="file" accept="image/*" onChange={onInput} hidden /></>}</div>{error && <p className="error">{error}</p>}<button className="primary-button" disabled={!file || loading} onClick={analyze}>{loading ? <><span className="spinner" /> Reading plant signals...</> : <><Zap size={17} /> Analyze image</>}</button><p className="privacy"><ShieldCheck size={14} /> Images are processed privately for this analysis</p></div><div className="result-column">{result ? <ResultPanel result={result} uploadedImage={preview} onReset={clear} /> : <EmptyState />}</div></section></>
}

function EmptyState() { return <div className="empty-state"><div className="empty-orbit"><Leaf size={29} /></div><p className="eyebrow">02 / YOUR FIELD NOTE</p><h2>Your result will<br /><em>appear here.</em></h2><p>Upload a clear, well-lit leaf image to see the predicted condition, confidence, and practical next steps.</p><div className="mini-features"><span><CheckCircle2 size={15} /> Confidence score</span><span><CheckCircle2 size={15} /> Top 3 signals</span><span><CheckCircle2 size={15} /> Care guidance</span></div></div> }

function ResultPanel({ result, uploadedImage, onReset }: { result: Analysis; uploadedImage: string; onReset: () => void }) {
  const confidence = Math.round(result.confidence * 100)
  const annotated = result.annotated_image_base64 ? `data:image/png;base64,${result.annotated_image_base64}` : uploadedImage || imageUrl(result.image_url)
  const lists = result.recommendations ? [['Management', result.recommendations.general_management], ['Prevention', result.recommendations.prevention], ['Monitoring', result.recommendations.monitoring]] as [string, string[]][] : []
  return <div className="result-panel"><div className="result-head"><div><p className="eyebrow">ANALYSIS COMPLETE</p><h2>{result.condition}</h2><p className="result-crop"><Leaf size={14} /> {result.crop} · MobileNetV2</p></div><span className={result.is_healthy ? 'health healthy' : 'health attention'}>{result.is_healthy ? 'Healthy signal' : 'Needs attention'}</span></div><div className="annotated-wrap">{annotated ? <img src={annotated} alt="Analyzed crop with detected disease areas" /> : <div className="no-image">No annotated image available</div>}</div><div className="confidence"><div><span>Model confidence</span><strong>{confidence}%</strong></div><div className="bar"><i style={{ width: `${confidence}%` }} /></div>{result.below_threshold && <p className="warning">Low confidence. Treat this reading as a prompt for closer inspection.</p>}</div><div className="predictions"><p className="eyebrow">TOP SIGNALS</p>{result.top_predictions.map((prediction, index) => <div className="prediction" key={prediction.condition}><span><b>{String(index + 1).padStart(2, '0')}</b>{prediction.condition}</span><strong>{Math.round(prediction.score * 100)}%</strong><div className="bar"><i style={{ width: `${prediction.score * 100}%` }} /></div></div>)}</div><div className="detection-summary"><div><p className="eyebrow">DETECTION SCAN</p><strong>{result.detections.length ? `${result.detections.length} area${result.detections.length === 1 ? '' : 's'} detected` : 'No significant disease detected'}</strong></div>{result.detection_severity && <span className="severity">{result.detection_severity.level} · {Math.round(result.detection_severity.ratio * 100)}%</span>}</div>{result.detections.length > 0 && <div className="detection-list">{result.detections.map((detection, index) => <div key={`${detection.class_name}-${index}`}><span>{detection.class_name}</span><strong>{Math.round(detection.confidence * 100)}%</strong></div>)}</div>}{result.recommendations && <div className="guidance"><p className="eyebrow">FIELD GUIDANCE</p><p>{result.recommendations.description}</p>{lists.map(([title, items]) => <div className="guidance-list" key={title}><strong>{title}</strong><ul>{items.map(item => <li key={item}><CheckCircle2 size={14} />{item}</li>)}</ul></div>)}</div>}<button className="outline-button another" onClick={onReset}>Analyze another image <ArrowRight size={15} /></button></div>
}

function HistoryView({ items, error, onSelect }: { items: Analysis[]; error: string; onSelect: (item: Analysis) => void }) { return <section className="page-section"><p className="eyebrow">FIELD LOG / PAST READINGS</p><h1>Analysis history</h1><p className="lede">A running record of every image you have checked.</p>{error && <p className="error">{error}</p>}{items.length ? <div className="history-list">{items.map(item => <button className="history-item" key={item.id} onClick={() => onSelect(item)}>{item.image_url ? <img className="history-thumb" src={imageUrl(item.image_url)} alt="" /> : <span className="history-icon"><Leaf size={18} /></span>}<span><strong>{item.condition}</strong><small>{item.crop} · {item.created_at ? new Date(item.created_at).toLocaleString() : 'Recent analysis'}</small></span><b>{Math.round(item.confidence * 100)}%</b><ArrowRight size={17} /></button>)}</div> : <div className="blank"><Clock3 size={27} /><h2>No field notes yet</h2><p>{error || 'Your completed analyses will appear here.'}</p></div>}</section> }
function DashboardView({ data, error }: { data: Dashboard | null; error: string }) { return <section className="page-section"><p className="eyebrow">FIELD OVERVIEW / SEASON TO DATE</p><h1>A clearer view of your crop health.</h1><p className="lede">Simple signals from your analysis history.</p>{error && <p className="error">{error}</p>}{data ? <div className="dashboard-grid"><div className="metric feature"><span>Total analyses</span><strong>{data.total_analyses}</strong><small>Images reviewed</small></div><div className="metric"><span>Healthy signals</span><strong>{data.healthy_count}</strong><small className="green">{data.total_analyses ? Math.round(data.healthy_count / data.total_analyses * 100) : 0}% of readings</small></div><div className="metric"><span>Need attention</span><strong>{data.attention_needed_count}</strong><small className="orange">Review recommended</small></div><div className="insight"><p className="eyebrow">MOST SEEN</p><h2>{data.most_detected_crop || 'No crop data yet'}</h2><p>{data.most_detected_condition || 'Complete an analysis to reveal recurring conditions.'}</p></div></div> : <div className="blank"><BarChart3 size={27} /><h2>Loading your overview</h2></div>}</section> }
function CropsView({ data, error }: { data: Crops | null; error: string }) { return <section className="page-section"><p className="eyebrow">REFERENCE / SUPPORTED CROPS</p><h1>Built for the plants<br /><em>you grow.</em></h1><p className="lede">The current model recognizes these crops and their common leaf conditions.</p>{error && <p className="error">{error}</p>}<div className="crop-grid">{data ? Object.entries(data).map(([crop, conditions], index) => <div className={`crop-card ${crop.toLowerCase()} crop-${index}`} key={crop}><span>{String(index + 1).padStart(2, '0')}</span><h2>{crop}</h2><p>{conditions.length} conditions · disease and healthy signals</p></div>) : <div className="blank">Loading supported crops</div>}</div></section> }


export default App
