import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { query } = await request.json()

    if (!query || typeof query !== "string") {
      return NextResponse.json({ error: "Invalid query parameter" }, { status: 400 })
    }

    const results = await searchWithFAISS(query)

    return NextResponse.json({ results })
  } catch (error) {
    console.error("[v0] Search error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

/**
 * Call the FastAPI FAISS search backend running on port 8000
 */
async function searchWithFAISS(query: string): Promise<any[]> {
  const BACKEND_URL = process.env.FAISS_BACKEND_URL || "http://localhost:8000"

  try {
    const response = await fetch(`${BACKEND_URL}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(`Backend returned ${response.status}: ${errorData.detail || "Unknown error"}`)
    }

    const data = await response.json()
    return data.results || []
  } catch (error) {
    console.error("[v0] Failed to call FastAPI backend:", error)
    throw error
  }
}
