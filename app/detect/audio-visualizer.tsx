"use client"

import { useEffect, useRef } from "react"

interface AudioVisualizerProps {
  isRecording: boolean
}

export function AudioVisualizer({ isRecording }: AudioVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!isRecording || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let animationFrameId: number

    const draw = () => {
      if (!ctx) return

      // Set canvas dimensions
      canvas.width = canvas.offsetWidth
      canvas.height = canvas.offsetHeight

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw visualization
      const barCount = 60
      const barWidth = canvas.width / barCount - 2

      for (let i = 0; i < barCount; i++) {
        // Generate random height for demonstration
        // In a real app, this would use actual audio data
        const height = Math.random() * canvas.height * 0.8

        const x = i * (barWidth + 2)
        const y = canvas.height - height

        // Apple-style visualization with blue color
        ctx.fillStyle = "#007AFF"
        ctx.strokeStyle = "#000000"
        ctx.lineWidth = 2

        // Draw bar with border
        ctx.beginPath()
        ctx.rect(x, y, barWidth, height)
        ctx.fill()
        ctx.stroke()
      }

      animationFrameId = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      cancelAnimationFrame(animationFrameId)
    }
  }, [isRecording])

  return <canvas ref={canvasRef} className="w-full h-full" width={300} height={100} />
}

