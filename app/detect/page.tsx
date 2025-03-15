"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import Link from "next/link"
import { ArrowLeft, Mic, Upload, Square, AlertCircle, CheckCircle2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AudioVisualizer } from "./audio-visualizer"

// API URL - change this to your server URL
const API_URL = "http://localhost:8000/analyze"

export default function DetectPage() {
  const [activeTab, setActiveTab] = useState("upload")
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisComplete, setAnalysisComplete] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<{
    score: number
    probability: string
    details: { name: string; score: number }[]
  } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Handle file upload
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setFileName(file.name)
      const url = URL.createObjectURL(file)
      setAudioUrl(url)
      setAudioBlob(file)
      setAnalysisComplete(false)
      setAnalysisResult(null)
      setError(null)
    }
  }

  // Handle recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" })
        const audioUrl = URL.createObjectURL(audioBlob)
        setAudioBlob(audioBlob)
        setAudioUrl(audioUrl)
        setFileName("recorded-audio.wav")
        setAnalysisComplete(false)
        setAnalysisResult(null)
        setError(null)

        // Stop all tracks in the stream
        stream.getTracks().forEach((track) => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error("Error accessing microphone:", error)
      setError("Error accessing microphone. Please make sure you have granted permission.")
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  // Analyze audio using the backend API
  const analyzeAudio = async () => {
    if (!audioBlob) return

    setIsAnalyzing(true)
    setAnalysisComplete(false)
    setError(null)

    try {
      // Create form data for the API request
      const formData = new FormData()
      formData.append("file", audioBlob)

      // Call the API
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      setAnalysisResult(data)
      setIsAnalyzing(false)
      setAnalysisComplete(true)
    } catch (error) {
      console.error("Error analyzing audio:", error)
      setError("Error analyzing audio. Please try again or check if the server is running.")
      setIsAnalyzing(false)
    }
  }

  // For demo purposes, if the API is not available
  const simulateAnalysis = () => {
    if (!audioBlob) return

    setIsAnalyzing(true)
    setAnalysisComplete(false)
    setError(null)

    // Simulate analysis delay
    setTimeout(() => {
      // Generate a random score between 0.1 and 0.9 for demonstration
      const fakeScore = Math.random() * 0.8 + 0.1

      // Create fake analysis details
      const details = [
        { name: "Frequency Patterns", score: Math.random() },
        { name: "Voice Naturalness", score: Math.random() },
        { name: "Background Noise", score: Math.random() },
        { name: "Temporal Consistency", score: Math.random() },
        { name: "Spectral Artifacts", score: Math.random() },
      ]

      // Determine probability text based on score
      let probability = "Low"
      if (fakeScore > 0.7) {
        probability = "High"
      } else if (fakeScore > 0.4) {
        probability = "Medium"
      }

      setAnalysisResult({
        score: fakeScore,
        probability,
        details,
      })

      setIsAnalyzing(false)
      setAnalysisComplete(true)
    }, 3000)
  }

  // Clean up URLs when component unmounts
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
      }
    }
  }, [audioUrl])

  return (
    <div className="min-h-screen bg-white">
      <div className="container max-w-4xl py-6 md:py-12">
        <div className="mb-6">
          <Link
            href="/"
            className="inline-flex items-center gap-1 text-base font-bold text-apple-blue hover:opacity-80"
          >
            <ArrowLeft className="h-5 w-5" />
            Back to WavSpoof
          </Link>
        </div>

        <div className="space-y-6">
          <div>
            <h1 className="text-4xl font-extrabold tracking-tight text-black">Audio Detection</h1>
            <p className="text-black mt-2 font-bold text-xl">
              Upload an audio file or record your voice to analyze for signs of AI generation.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2 p-1 bg-apple-gray-6 rounded-xl border-2 border-black">
              <TabsTrigger
                value="upload"
                className={`${activeTab === "upload" ? "bg-white text-apple-blue shadow-md border-2 border-black" : ""} transition-all duration-300 rounded-lg font-extrabold text-lg py-3`}
              >
                Upload Audio
              </TabsTrigger>
              <TabsTrigger
                value="record"
                className={`${activeTab === "record" ? "bg-white text-apple-blue shadow-md border-2 border-black" : ""} transition-all duration-300 rounded-lg font-extrabold text-lg py-3`}
              >
                Record Audio
              </TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="space-y-4">
              <Card className="border-2 border-black rounded-xl shadow-md">
                <CardHeader>
                  <CardTitle className="text-black font-extrabold text-2xl">Upload Audio File</CardTitle>
                  <CardDescription className="text-black font-bold text-lg">
                    Select an audio file from your device to analyze.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col items-center justify-center gap-4">
                    <div
                      className="border-2 border-dashed border-black rounded-xl p-12 w-full flex flex-col items-center justify-center cursor-pointer hover:bg-apple-gray-6 transition-colors"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload className="h-16 w-16 text-apple-blue mb-4" />
                      <p className="text-base text-center text-black font-bold">
                        Click to browse or drag and drop your audio file here
                      </p>
                      <p className="text-sm text-center text-black mt-2 font-bold">
                        Supports WAV, MP3, M4A, and OGG files
                      </p>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="audio/*"
                        className="hidden"
                        onChange={handleFileChange}
                      />
                    </div>

                    {fileName && (
                      <div className="text-base text-black font-bold">
                        Selected file: <span className="font-extrabold">{fileName}</span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="record" className="space-y-4">
              <Card className="border-2 border-black rounded-xl shadow-md">
                <CardHeader>
                  <CardTitle className="text-black font-extrabold text-2xl">Record Your Voice</CardTitle>
                  <CardDescription className="text-black font-bold text-lg">
                    Record audio directly from your microphone.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col items-center justify-center gap-6">
                    <div className="w-full h-40 bg-apple-gray-6 rounded-xl flex items-center justify-center border-2 border-black">
                      {isRecording ? (
                        <div className="w-full h-full">
                          <AudioVisualizer isRecording={isRecording} />
                        </div>
                      ) : (
                        <p className="text-black font-bold text-xl">
                          {audioUrl ? "Recording complete" : "Ready to record"}
                        </p>
                      )}
                    </div>

                    <div className="flex gap-4">
                      {!isRecording ? (
                        <Button
                          onClick={startRecording}
                          disabled={isRecording}
                          className="gap-2 bg-apple-blue hover:bg-apple-blue-dark text-white font-extrabold text-lg py-6 px-6"
                        >
                          <Mic className="h-6 w-6" />
                          Start Recording
                        </Button>
                      ) : (
                        <Button
                          onClick={stopRecording}
                          variant="destructive"
                          className="gap-2 bg-apple-red hover:opacity-90 text-white font-extrabold text-lg py-6 px-6"
                        >
                          <Square className="h-6 w-6" />
                          Stop Recording
                        </Button>
                      )}
                    </div>

                    {fileName && !isRecording && (
                      <div className="text-base text-black font-bold">
                        Recording saved as: <span className="font-extrabold">{fileName}</span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {error && (
            <div className="bg-apple-red/10 border-2 border-apple-red rounded-xl p-4 text-apple-red font-bold">
              {error}
            </div>
          )}

          {audioUrl && !isRecording && (
            <Card className="border-2 border-black rounded-xl shadow-md">
              <CardHeader>
                <CardTitle className="text-black font-extrabold text-2xl">Audio Preview</CardTitle>
                <CardDescription className="text-black font-bold text-lg">
                  Listen to your audio before analysis.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <audio src={audioUrl} controls className="w-full h-12" />
              </CardContent>
              <CardFooter>
                <Button
                  onClick={analyzeAudio}
                  disabled={isAnalyzing}
                  className="w-full bg-apple-blue hover:bg-apple-blue-dark text-white font-extrabold text-lg py-6"
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze Audio"}
                </Button>
              </CardFooter>
            </Card>
          )}

          {isAnalyzing && (
            <Card className="border-2 border-black rounded-xl shadow-md">
              <CardHeader>
                <CardTitle className="text-black font-extrabold text-2xl">Analyzing Audio</CardTitle>
                <CardDescription className="text-black font-bold text-lg">
                  Please wait while we process your audio.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="relative pt-1">
                    <div className="overflow-hidden h-4 text-xs flex rounded-full bg-apple-gray-5 border-2 border-black">
                      <div
                        style={{ width: "75%" }}
                        className="animate-pulse shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-apple-blue"
                      ></div>
                    </div>
                  </div>
                  <p className="text-base text-center text-black font-bold">
                    Analyzing frequency patterns, voice characteristics, and other indicators...
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {analysisComplete && analysisResult && (
            <Card className="border-2 border-black rounded-xl shadow-md">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-black font-extrabold text-2xl">
                  Analysis Results
                  {analysisResult.score > 0.7 ? (
                    <AlertCircle className="h-8 w-8 text-apple-red" />
                  ) : (
                    <CheckCircle2 className="h-8 w-8 text-apple-green" />
                  )}
                </CardTitle>
                <CardDescription className="text-black font-bold text-lg">
                  Our analysis of the provided audio sample.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <h3 className="font-extrabold text-black text-xl">
                        AI Generation Probability: {analysisResult.probability}
                      </h3>
                      <span
                        className={`text-base font-extrabold px-3 py-1 rounded-full border-2 ${
                          analysisResult.score > 0.7
                            ? "bg-apple-red/10 text-apple-red border-apple-red"
                            : analysisResult.score > 0.4
                              ? "bg-apple-orange/10 text-apple-orange border-apple-orange"
                              : "bg-apple-green/10 text-apple-green border-apple-green"
                        }`}
                      >
                        {Math.round(analysisResult.score * 100)}%
                      </span>
                    </div>
                    <div className="relative pt-1">
                      <div className="overflow-hidden h-4 text-xs flex rounded-full bg-apple-gray-5 border-2 border-black">
                        <div
                          style={{ width: `${Math.round(analysisResult.score * 100)}%` }}
                          className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${
                            analysisResult.score > 0.7
                              ? "bg-apple-red"
                              : analysisResult.score > 0.4
                                ? "bg-apple-orange"
                                : "bg-apple-green"
                          }`}
                        ></div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-extrabold mb-3 text-black text-xl">Detailed Analysis</h3>
                    <div className="space-y-4">
                      {analysisResult.details.map((detail, index) => (
                        <div key={index} className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-base text-black font-bold">{detail.name}</span>
                            <span
                              className={`text-sm font-extrabold px-2 py-1 rounded-full border ${
                                detail.score > 0.7
                                  ? "bg-apple-red/10 text-apple-red border-apple-red"
                                  : detail.score > 0.4
                                    ? "bg-apple-orange/10 text-apple-orange border-apple-orange"
                                    : "bg-apple-green/10 text-apple-green border-apple-green"
                              }`}
                            >
                              {Math.round(detail.score * 100)}%
                            </span>
                          </div>
                          <div className="relative pt-1">
                            <div className="overflow-hidden h-3 text-xs flex rounded-full bg-apple-gray-5 border border-black">
                              <div
                                style={{ width: `${Math.round(detail.score * 100)}%` }}
                                className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${
                                  detail.score > 0.7
                                    ? "bg-apple-red"
                                    : detail.score > 0.4
                                      ? "bg-apple-orange"
                                      : "bg-apple-green"
                                }`}
                              ></div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div
                    className={`rounded-xl p-4 border-2 ${
                      analysisResult.score > 0.7
                        ? "bg-apple-red/5 border-apple-red"
                        : analysisResult.score > 0.4
                          ? "bg-apple-orange/5 border-apple-orange"
                          : "bg-apple-green/5 border-apple-green"
                    }`}
                  >
                    <h3 className="font-extrabold mb-2 text-black text-xl">Conclusion</h3>
                    <p className="text-base text-black font-bold">
                      {analysisResult.score > 0.7
                        ? "This audio sample shows strong indicators of AI generation or manipulation. The unnatural patterns in frequency distribution and voice characteristics suggest synthetic origin."
                        : analysisResult.score > 0.4
                          ? "This audio sample shows some indicators that may suggest AI involvement, but the evidence is not conclusive. Some aspects appear natural while others show potential manipulation."
                          : "This audio sample appears to be authentic human speech with high probability. We detected minimal indicators of AI generation or manipulation."}
                    </p>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button
                  variant="outline"
                  onClick={() => {
                    setAudioBlob(null)
                    setAudioUrl(null)
                    setFileName(null)
                    setAnalysisComplete(false)
                    setAnalysisResult(null)
                  }}
                  className="border-2 border-black text-black hover:bg-apple-gray-6 font-bold text-base"
                >
                  Start Over
                </Button>
                <Button
                  onClick={analyzeAudio}
                  className="bg-apple-blue hover:bg-apple-blue-dark text-white font-extrabold text-base"
                >
                  Analyze Again
                </Button>
              </CardFooter>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

