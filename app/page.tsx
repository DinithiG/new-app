import Link from "next/link"
import { ArrowRight, Shield, AudioWaveformIcon as Waveform, Upload, Mic } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-white">
      <header className="px-4 lg:px-6 h-16 flex items-center border-b-2 border-black">
        <Link href="/" className="flex items-center gap-2 font-extrabold text-2xl">
          <Shield className="h-8 w-8 text-apple-blue" />
          <span className="text-black">WavSpoof</span>
        </Link>
        <nav className="ml-auto flex gap-4 sm:gap-6">
          <Link href="#features" className="text-base font-bold text-black hover:text-apple-blue transition-colors">
            Features
          </Link>
          <Link href="#how-it-works" className="text-base font-bold text-black hover:text-apple-blue transition-colors">
            How It Works
          </Link>
          <Link href="/detect" className="text-base font-bold text-black hover:text-apple-blue transition-colors">
            Try It
          </Link>
        </nav>
      </header>
      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48">
          <div className="container px-4 md:px-6">
            <div className="grid gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_600px]">
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h1 className="text-4xl font-extrabold tracking-tight sm:text-6xl xl:text-7xl/none text-black">
                    Detect AI-Generated Audio
                  </h1>
                  <p className="max-w-[600px] text-black md:text-2xl font-bold">
                    Our advanced technology helps you identify synthetic or manipulated audio with high accuracy.
                  </p>
                </div>
                <div className="flex flex-col gap-2 min-[400px]:flex-row">
                  <Link href="/detect">
                    <Button
                      size="lg"
                      className="gap-1.5 bg-apple-blue hover:bg-apple-blue-dark text-white text-lg font-extrabold h-14"
                    >
                      Try It Now
                      <ArrowRight className="h-5 w-5" />
                    </Button>
                  </Link>
                  <Link href="#how-it-works">
                    <Button
                      size="lg"
                      variant="outline"
                      className="border-2 border-black text-black hover:bg-apple-gray-6 text-lg font-bold h-14"
                    >
                      Learn More
                    </Button>
                  </Link>
                </div>
              </div>
              <div className="flex items-center justify-center">
                <div className="relative w-full h-[300px] md:h-[400px] bg-apple-gray-6 rounded-2xl overflow-hidden border-2 border-black">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Waveform className="h-32 w-32 text-apple-blue" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section id="features" className="w-full py-12 md:py-24 lg:py-32 bg-apple-gray-6">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <div className="inline-block rounded-full bg-apple-blue/10 px-4 py-2 text-base font-extrabold text-apple-blue border-2 border-apple-blue">
                  Features
                </div>
                <h2 className="text-3xl font-extrabold tracking-tight sm:text-5xl text-black">How WavSpoof Works</h2>
                <p className="max-w-[900px] text-black md:text-2xl/relaxed lg:text-xl/relaxed xl:text-2xl/relaxed font-bold">
                  Our technology analyzes multiple aspects of audio to detect signs of AI generation or manipulation.
                </p>
              </div>
            </div>
            <div className="mx-auto grid max-w-5xl items-center gap-6 py-12 lg:grid-cols-3 lg:gap-12">
              <div className="flex flex-col justify-center space-y-4">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-apple-blue text-white border-2 border-black">
                  <Waveform className="h-8 w-8" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-2xl font-extrabold text-black">Spectral Analysis</h3>
                  <p className="text-black font-bold text-lg">
                    We analyze frequency patterns that are characteristic of AI-generated audio.
                  </p>
                </div>
              </div>
              <div className="flex flex-col justify-center space-y-4">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-apple-green text-white border-2 border-black">
                  <Mic className="h-8 w-8" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-2xl font-extrabold text-black">Voice Characteristics</h3>
                  <p className="text-black font-bold text-lg">
                    Our system detects unnatural patterns in voice modulation and articulation.
                  </p>
                </div>
              </div>
              <div className="flex flex-col justify-center space-y-4">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-apple-orange text-white border-2 border-black">
                  <Upload className="h-8 w-8" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-2xl font-extrabold text-black">Easy to Use</h3>
                  <p className="text-black font-bold text-lg">
                    Simply upload an audio file or record directly in your browser to get instant results.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
      <footer className="flex flex-col gap-2 sm:flex-row py-6 w-full shrink-0 items-center px-4 md:px-6 border-t-2 border-black">
        <p className="text-sm text-black font-bold">© {new Date().getFullYear()} WavSpoof. All rights reserved.</p>
        <nav className="sm:ml-auto flex gap-4 sm:gap-6">
          <Link
            href="#"
            className="text-sm text-black font-bold hover:text-apple-blue hover:underline underline-offset-4"
          >
            Terms of Service
          </Link>
          <Link
            href="#"
            className="text-sm text-black font-bold hover:text-apple-blue hover:underline underline-offset-4"
          >
            Privacy
          </Link>
        </nav>
      </footer>
    </div>
  )
}

