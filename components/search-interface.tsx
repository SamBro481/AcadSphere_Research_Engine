"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { Search, Sparkles, FileText, Calendar, User, GraduationCap, Mic, MicOff } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface Paper {
  title: string
  authors?: string[]
  author?: string
  abstract: string
  year?: number
  venue?: string
  score: number
  base_score?: number
  summary?: string
}

export function SearchInterface() {
  const [query, setQuery] = useState("")
  const [isSearching, setIsSearching] = useState(false)
  const [results, setResults] = useState<Paper[]>([])
  const [hasSearched, setHasSearched] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isListening, setIsListening] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState("Fetching relevant Research Papers for you")
  const recognitionRef = useRef<any>(null)

  useEffect(() => {
    if (typeof window !== "undefined" && ("SpeechRecognition" in window || "webkitSpeechRecognition" in window)) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = false
      recognitionRef.current.interimResults = false

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript
        setQuery(transcript)
        setIsListening(false)
      }

      recognitionRef.current.onerror = () => {
        setIsListening(false)
      }

      recognitionRef.current.onend = () => {
        setIsListening(false)
      }
    }
  }, [])

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop()
    } else {
      setIsListening(true)
      recognitionRef.current?.start()
    }
  }

  const handleReload = () => {
    window.location.reload()
  }

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setIsSearching(true)
    setHasSearched(true)
    setError(null)
    setResults([])
    setLoadingMessage("Fetching relevant Research Papers for you")

    const messageTimer = setTimeout(() => {
      setLoadingMessage("Generating paper summaries")
    }, 3500)

    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Search failed: ${response.statusText}`)
      }

      const data = await response.json()

      const validResults = (data.results || []).filter((p: any) => {
        return p && typeof p === "object" && p.title && p.abstract
      })

      setResults(validResults)
    } catch (err) {
      console.error("[v0] Search error:", err)
      setError(err instanceof Error ? err.message : "An error occurred")
      setResults([])
    } finally {
      clearTimeout(messageTimer)
      setIsSearching(false)
    }
  }

  return (
    <div className="min-h-screen bg-background flex flex-col font-sans selection:bg-primary/20">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <button onClick={handleReload} className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <GraduationCap className="w-6 h-6 text-primary" />
            <h1 className="text-lg font-normal tracking-tight text-foreground">AcadSphere</h1>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12 flex-1">
        {!hasSearched && (
          <div className="max-w-3xl mx-auto text-center mb-12 space-y-8">
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/50 text-accent-foreground text-xs border border-border">
                <Sparkles className="w-4 h-4" />
                <span>Powered by AI</span>
              </div>
              <h2 className="text-3xl md:text-4xl font-normal text-foreground text-balance leading-tight tracking-tight">
                {"Search for your next Research Paper"}
              </h2>
            </div>

            <form onSubmit={handleSearch}>
              <div className="relative group">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground group-focus-within:text-primary transition-colors" />
                <Input
                  type="text"
                  placeholder="Ask Anything"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch(e)}
                  className="pl-12 pr-12 h-14 text-base bg-card border-border rounded-xl shadow-sm focus-visible:ring-1 focus-visible:ring-primary/50 transition-all font-normal"
                />
                <button
                  type="button"
                  onClick={toggleListening}
                  className={`absolute right-4 top-1/2 -translate-y-1/2 p-2 rounded-full transition-all ${
                    isListening
                      ? "bg-primary text-primary-foreground animate-pulse"
                      : "text-muted-foreground hover:bg-accent"
                  }`}
                >
                  {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                </button>
              </div>
            </form>
          </div>
        )}

        {isSearching && (
          <div className="max-w-3xl mx-auto text-center py-20">
            <div className="inline-flex flex-col items-center gap-6">
              <div className="relative w-16 h-16">
                <div className="absolute inset-0 rounded-full border-4 border-muted"></div>
                <div className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent animate-spin"></div>
              </div>
              <p className="text-lg text-muted-foreground">{loadingMessage}</p>
            </div>
          </div>
        )}

        {/* Results */}
        {hasSearched && !isSearching && (
          <div className="max-w-4xl mx-auto">
            {error ? (
              <Card className="p-12 text-center bg-card border-destructive/50">
                <div className="max-w-md mx-auto space-y-3">
                  <div className="w-16 h-16 rounded-full bg-destructive/10 flex items-center justify-center mx-auto">
                    <Search className="w-8 h-8 text-destructive" />
                  </div>
                  <h3 className="text-xl font-semibold text-foreground">{"Search Error"}</h3>
                  <p className="text-muted-foreground">{error}</p>
                  <p className="text-sm text-muted-foreground">
                    {"Make sure the FastAPI backend is running on port 8000"}
                  </p>
                </div>
              </Card>
            ) : results.length > 0 ? (
              <>
                <div className="mb-6">
                  <p className="text-sm text-muted-foreground">
                    {"Found "}
                    <span className="font-semibold text-foreground">{results.length}</span>
                    {" relevant papers"}
                  </p>
                </div>
                <div className="space-y-4">
                  {results.map((paper, idx) => {
                    if (!paper) return null

                    return (
                      <Card
                        key={idx}
                        className="p-6 hover:shadow-lg transition-all duration-200 bg-card border-border animate-in fade-in slide-in-from-bottom-4"
                        style={{ animationDelay: `${idx * 50}ms` }}
                      >
                        <div className="space-y-4">
                          <div className="flex items-start justify-between gap-4">
                            <h3 className="text-xl font-semibold text-foreground leading-tight flex-1">
                              {paper.title || "Untitled Research Paper"}
                            </h3>
                            <Badge variant="secondary" className="shrink-0">
                              {Math.round((paper.score || 0) * 100)}
                              {"% match"}
                            </Badge>
                          </div>

                          {(paper.authors || paper.author) && (
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <User className="w-4 h-4" />
                              <span>{paper.authors ? paper.authors.join(", ") : paper.author}</span>
                            </div>
                          )}

                          <p className="text-muted-foreground leading-relaxed line-clamp-3">{paper.abstract}</p>

                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            {paper.year && (
                              <div className="flex items-center gap-1.5">
                                <Calendar className="w-4 h-4" />
                                <span>{paper.year}</span>
                              </div>
                            )}
                            {paper.venue && (
                              <div className="flex items-center gap-1.5">
                                <FileText className="w-4 h-4" />
                                <span>{paper.venue}</span>
                              </div>
                            )}
                          </div>

                          {paper.summary && (
                            <div className="mt-4 p-6 bg-accent/10 rounded-xl border border-border/50 space-y-3">
                              <div className="flex items-center gap-2 mb-3">
                                <Sparkles className="w-4 h-4 text-primary" />
                                <h4 className="text-xs font-semibold tracking-widest uppercase text-muted-foreground">
                                  AI Generated Summary
                                </h4>
                              </div>
                              <div className="prose prose-sm max-w-none leading-relaxed font-normal space-y-3">
                                {paper.summary.split("\n").map((line, i) => {
                                  const cleanLine = line
                                    .replace(/^\d+\.\s*/, "")
                                    .replace(/\*\*/g, "")
                                    .trim()

                                  if (!cleanLine) return null

                                  if (
                                    cleanLine.endsWith(":") ||
                                    cleanLine.toLowerCase().includes("summary") ||
                                    cleanLine.toLowerCase().includes("key findings") ||
                                    cleanLine.toLowerCase().includes("methods") ||
                                    cleanLine.toLowerCase().includes("results") ||
                                    cleanLine.toLowerCase().includes("why you should read")
                                  ) {
                                    return (
                                      <h5 key={i} className="text-base font-semibold text-white mt-4 mb-2 first:mt-0">
                                        {cleanLine.replace(/:$/, "")}
                                      </h5>
                                    )
                                  }

                                  if (cleanLine.startsWith("-") || cleanLine.startsWith("•")) {
                                    return (
                                      <div key={i} className="flex gap-3 ml-1 mb-2">
                                        <span className="text-primary mt-2 w-1.5 h-1.5 rounded-full bg-primary shrink-0" />
                                        <p className="text-muted-foreground">{cleanLine.replace(/^[-•]\s*/, "")}</p>
                                      </div>
                                    )
                                  }

                                  return (
                                    <p key={i} className="text-muted-foreground mb-3 last:mb-0">
                                      {cleanLine}
                                    </p>
                                  )
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      </Card>
                    )
                  })}
                </div>

                <div className="sticky bottom-8 max-w-3xl mx-auto mt-8">
                  <form onSubmit={handleSearch}>
                    <div className="relative group">
                      <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground group-focus-within:text-primary transition-colors" />
                      <Input
                        type="text"
                        placeholder="Search another Research Topic"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSearch(e)}
                        className="pl-12 pr-12 h-14 text-base bg-card border-border rounded-xl shadow-lg focus-visible:ring-1 focus-visible:ring-primary/50 transition-all font-normal"
                      />
                      <button
                        type="button"
                        onClick={toggleListening}
                        className={`absolute right-4 top-1/2 -translate-y-1/2 p-2 rounded-full transition-all ${
                          isListening
                            ? "bg-primary text-primary-foreground animate-pulse"
                            : "text-muted-foreground hover:bg-accent"
                        }`}
                      >
                        {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                      </button>
                    </div>
                  </form>
                </div>
              </>
            ) : results.length === 0 && !error ? (
              <Card className="p-12 text-center bg-card border-border">
                <div className="max-w-md mx-auto space-y-3">
                  <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto">
                    <Search className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <h3 className="text-xl font-semibold text-foreground">{"No results found"}</h3>
                  <p className="text-muted-foreground">
                    {"Try adjusting your search query or using different keywords."}
                  </p>
                </div>
              </Card>
            ) : null}
          </div>
        )}
      </div>

      <footer className="border-t border-border bg-card/40 backdrop-blur-sm mt-auto">
        <div className="container mx-auto px-4 py-6">
          <div className="text-center space-y-2">
            <p className="text-sm text-muted-foreground">Explore More</p>
            <a
              href="https://www.samsportfolio.xyz"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted-foreground/70 hover:text-primary transition-colors duration-200 inline-block"
            >
              www.samsportfolio.xyz
            </a>
          </div>
        </div>
      </footer>
    </div>
  )
}
