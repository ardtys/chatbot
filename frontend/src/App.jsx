import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import { Send, FileText, ChevronRight, Plus, Trash2, Menu, X } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || ''

const api = {
  query: (question, chatHistory = []) =>
    fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, chat_history: chatHistory }),
    }).then(r => {
      if (!r.ok) throw new Error('Gagal memproses')
      return r.json()
    }),
  health: () => fetch(`${API_BASE}/api/health`).then(r => r.json()).catch(() => null),
}

function Sources({ data }) {
  const [open, setOpen] = useState(false)
  if (!data?.length) return null

  return (
    <div className="mt-4 pt-4 border-t border-neutral-100">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 text-[13px] text-neutral-400 hover:text-neutral-600"
      >
        <FileText size={14} />
        <span>Lihat {data.length} sumber</span>
        <ChevronRight size={14} className={`transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      {open && (
        <div className="mt-3 space-y-2">
          {data.map((s, i) => (
            <div key={i} className="text-[13px] p-3 bg-neutral-50 rounded-lg border border-neutral-100">
              <div className="font-medium text-neutral-700">{s.source}</div>
              <div className="text-neutral-400 text-[12px] mb-2">Halaman {s.page}</div>
              <div className="text-neutral-500 leading-relaxed">{s.content.slice(0, 200)}...</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Message({ data, isLast }) {
  const isUser = data.role === 'user'

  if (isUser) {
    return (
      <div className={`flex justify-end ${isLast ? 'animate-in' : ''}`}>
        <div className="max-w-[90%] sm:max-w-[85%] bg-neutral-900 text-white px-3 sm:px-4 py-2.5 sm:py-3 rounded-2xl rounded-br-sm">
          <p className="text-[14px] sm:text-[15px] leading-relaxed">{data.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`${isLast ? 'animate-in' : ''}`}>
      <div className="prose-answer">
        <ReactMarkdown>{data.content}</ReactMarkdown>
      </div>
      <Sources data={data.sources} />
    </div>
  )
}

function Typing() {
  return (
    <div className="flex items-center gap-1 py-4">
      <span className="w-2 h-2 bg-neutral-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
      <span className="w-2 h-2 bg-neutral-300 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
      <span className="w-2 h-2 bg-neutral-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
    </div>
  )
}

export default function App() {
  const [chats, setChats] = useState([{ id: 1, title: 'Percakapan Baru', messages: [] }])
  const [activeChat, setActiveChat] = useState(1)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState(null)
  const [history, setHistory] = useState([])
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const scrollRef = useRef(null)
  const inputRef = useRef(null)

  const currentChat = chats.find(c => c.id === activeChat) || chats[0]
  const messages = currentChat?.messages || []

  useEffect(() => {
    api.health().then(setHealth)
    const i = setInterval(() => api.health().then(setHealth), 30000)
    return () => clearInterval(i)
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, loading])

  const send = useCallback(async (text) => {
    const q = text || input.trim()
    if (!q || loading) return

    setInput('')
    setChats(prev => prev.map(c =>
      c.id === activeChat
        ? { ...c, messages: [...c.messages, { role: 'user', content: q }], title: c.messages.length === 0 ? q.slice(0, 30) : c.title }
        : c
    ))
    setLoading(true)

    try {
      const res = await api.query(q, history)
      setChats(prev => prev.map(c =>
        c.id === activeChat
          ? { ...c, messages: [...c.messages, { role: 'assistant', content: res.answer, sources: res.sources }] }
          : c
      ))
      setHistory(prev => [...prev, [q, res.answer]])
    } catch {
      setChats(prev => prev.map(c =>
        c.id === activeChat
          ? { ...c, messages: [...c.messages, { role: 'assistant', content: 'Maaf, terjadi kesalahan. Silakan coba lagi.', sources: [] }] }
          : c
      ))
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }, [input, loading, history, activeChat])

  const newChat = () => {
    const id = Date.now()
    setChats(prev => [...prev, { id, title: 'Percakapan Baru', messages: [] }])
    setActiveChat(id)
    setHistory([])
  }

  const deleteChat = (id) => {
    if (chats.length === 1) {
      setChats([{ id: Date.now(), title: 'Percakapan Baru', messages: [] }])
      setActiveChat(chats[0].id)
    } else {
      setChats(prev => prev.filter(c => c.id !== id))
      if (activeChat === id) setActiveChat(chats[0].id)
    }
    setHistory([])
  }

  const suggestions = [
    'Apa saja jenis cuti yang tersedia?',
    'Bagaimana cara mengajukan pinjaman karyawan?',
    'Prosedur klaim medical rawat jalan',
    'Ketentuan work from home',
  ]

  const isOnline = health?.groq_configured || health?.ollama_connected

  return (
    <div className="h-screen flex bg-white">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-72 bg-neutral-950 flex flex-col transform transition-transform duration-300 ease-in-out
        lg:static lg:w-64 lg:translate-x-0
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="p-4">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-white rounded-lg flex items-center justify-center">
                <span className="text-lg font-bold text-neutral-900">B</span>
              </div>
              <div>
                <div className="text-white font-semibold text-[15px]">Basis</div>
                <div className="text-neutral-500 text-[11px]">Company Assistant</div>
              </div>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-2 text-neutral-400 hover:text-white"
            >
              <X size={20} />
            </button>
          </div>
          <button
            onClick={() => { newChat(); setSidebarOpen(false) }}
            className="w-full flex items-center gap-2 px-3 py-2.5 bg-neutral-800 hover:bg-neutral-700 text-white text-[13px] font-medium rounded-lg transition-colors"
          >
            <Plus size={16} />
            Percakapan Baru
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2">
          {chats.map(chat => (
            <div
              key={chat.id}
              className={`group flex items-center gap-2 px-3 py-2.5 mb-1 rounded-lg cursor-pointer transition-colors ${
                activeChat === chat.id ? 'bg-neutral-800' : 'hover:bg-neutral-900'
              }`}
              onClick={() => { setActiveChat(chat.id); setHistory([]); setSidebarOpen(false) }}
            >
              <span className="flex-1 text-[13px] text-neutral-300 truncate">{chat.title}</span>
              <button
                onClick={(e) => { e.stopPropagation(); deleteChat(chat.id) }}
                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-neutral-700 rounded transition-all"
              >
                <Trash2 size={14} className="text-neutral-500" />
              </button>
            </div>
          ))}
        </div>

        <div className="p-4 border-t border-neutral-800">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-neutral-600'}`} />
            <span className="text-[12px] text-neutral-500">
              {isOnline ? 'Online' : 'Connecting...'}
            </span>
          </div>
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile header */}
        <div className="lg:hidden flex items-center gap-3 px-4 py-3 border-b border-neutral-100">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 -ml-2 text-neutral-600 hover:text-neutral-900"
          >
            <Menu size={22} />
          </button>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 bg-neutral-900 rounded-lg flex items-center justify-center">
              <span className="text-sm font-bold text-white">B</span>
            </div>
            <span className="font-semibold text-neutral-900">Basis</span>
          </div>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="max-w-2xl mx-auto px-4 sm:px-6 py-6 sm:py-8">
            {messages.length === 0 ? (
              <div className="pt-8 sm:pt-16">
                <h1 className="text-2xl sm:text-3xl font-semibold text-neutral-900 mb-2">Hai, ada yang bisa dibantu?</h1>
                <p className="text-neutral-500 mb-8 sm:mb-10 text-sm sm:text-base">Tanyakan apapun seputar kebijakan dan prosedur perusahaan.</p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3">
                  {suggestions.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => send(s)}
                      className="text-left p-3 sm:p-4 border border-neutral-200 rounded-xl text-[13px] sm:text-[14px] text-neutral-600 hover:border-neutral-300 hover:bg-neutral-50 transition-all active:bg-neutral-100"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-4 sm:space-y-6">
                {messages.map((m, i) => (
                  <Message key={i} data={m} isLast={i === messages.length - 1 && !loading} />
                ))}
                {loading && <Typing />}
              </div>
            )}
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-neutral-100 bg-white safe-bottom">
          <div className="max-w-2xl mx-auto px-4 sm:px-6 py-3 sm:py-4">
            <div className="flex items-end gap-2 sm:gap-3">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }}}
                placeholder="Tulis pertanyaan..."
                rows={1}
                className="flex-1 px-3 sm:px-4 py-3 bg-neutral-100 border-0 rounded-xl text-[15px] text-neutral-800 placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-neutral-200 resize-none"
                style={{ minHeight: '48px', maxHeight: '120px' }}
                onInput={(e) => { e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px' }}
              />
              <button
                onClick={() => send()}
                disabled={!input.trim() || loading}
                className="w-11 h-11 sm:w-12 sm:h-12 bg-neutral-900 hover:bg-neutral-800 disabled:bg-neutral-200 rounded-xl flex items-center justify-center transition-colors flex-shrink-0"
              >
                <Send size={18} className="text-white" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
