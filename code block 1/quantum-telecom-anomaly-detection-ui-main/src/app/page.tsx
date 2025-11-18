"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Network, Brain, Zap, Shield, Activity, CheckCircle2, ArrowRight } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="mb-16 text-center">
          <Badge className="mb-4 bg-blue-500/20 text-blue-300 hover:bg-blue-500/30">
            Quantum Machine Learning
          </Badge>
          <h1 className="mb-6 text-5xl font-bold tracking-tight md:text-6xl lg:text-7xl">
            Network Anomaly Detection
          </h1>
          <p className="mx-auto mb-8 max-w-2xl text-xl text-slate-300">
            AI-powered telecom security using hybrid quantum neural networks and quantum support vector machines
          </p>
          <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Link href="/network-anomaly">
              <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
                <Activity className="mr-2 h-5 w-5" />
                Launch Dashboard
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <Button size="lg" variant="outline" className="border-slate-700 bg-slate-900/50 hover:bg-slate-800">
              <Brain className="mr-2 h-5 w-5" />
              View Documentation
            </Button>
          </div>
        </div>

        {/* Features Grid */}
        <div className="mb-20 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          <Card className="border-blue-500/20 bg-slate-900/50 backdrop-blur">
            <CardHeader>
              <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-blue-500/20">
                <Brain className="h-6 w-6 text-blue-400" />
              </div>
              <CardTitle className="text-white">Hybrid Quantum Neural Network</CardTitle>
              <CardDescription className="text-slate-400">
                Variational quantum circuits with classical preprocessing for pattern recognition
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-slate-300">
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-400" />
                  4-qubit quantum circuit
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-400" />
                  94.7% training accuracy
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-400" />
                  Real-time inference
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="border-purple-500/20 bg-slate-900/50 backdrop-blur">
            <CardHeader>
              <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-purple-500/20">
                <Zap className="h-6 w-6 text-purple-400" />
              </div>
              <CardTitle className="text-white">Quantum Support Vector Machine</CardTitle>
              <CardDescription className="text-slate-400">
                Quantum kernel methods for high-dimensional classification
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-slate-300">
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-400" />
                  Quantum kernel mapping
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-400" />
                  91.3% classification accuracy
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-400" />
                  47 support vectors
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="border-green-500/20 bg-slate-900/50 backdrop-blur">
            <CardHeader>
              <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-green-500/20">
                <Shield className="h-6 w-6 text-green-400" />
              </div>
              <CardTitle className="text-white">Real-Time Monitoring</CardTitle>
              <CardDescription className="text-slate-400">
                Continuous network traffic analysis and threat detection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-slate-300">
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-400" />
                  Live metric tracking
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-400" />
                  Instant anomaly alerts
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-400" />
                  Batch data analysis
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>

        {/* Tech Stack */}
        <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-center text-white">Quantum ML Pipeline</CardTitle>
            <CardDescription className="text-center text-slate-400">
              End-to-end architecture for telecom network security
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center justify-center gap-6">
              <div className="flex flex-col items-center">
                <div className="mb-2 flex h-16 w-16 items-center justify-center rounded-full bg-blue-500/20">
                  <Network className="h-8 w-8 text-blue-400" />
                </div>
                <span className="text-sm text-slate-300">Network Data</span>
              </div>
              
              <ArrowRight className="h-6 w-6 text-slate-600" />
              
              <div className="flex flex-col items-center">
                <div className="mb-2 flex h-16 w-16 items-center justify-center rounded-full bg-purple-500/20">
                  <span className="text-2xl">⚙️</span>
                </div>
                <span className="text-sm text-slate-300">Preprocessing</span>
              </div>
              
              <ArrowRight className="h-6 w-6 text-slate-600" />
              
              <div className="flex flex-col items-center">
                <div className="mb-2 flex h-16 w-16 items-center justify-center rounded-full bg-amber-500/20">
                  <span className="text-2xl">⚛️</span>
                </div>
                <span className="text-sm text-slate-300">Quantum Layer</span>
              </div>
              
              <ArrowRight className="h-6 w-6 text-slate-600" />
              
              <div className="flex flex-col items-center">
                <div className="mb-2 flex h-16 w-16 items-center justify-center rounded-full bg-green-500/20">
                  <span className="text-2xl">🎯</span>
                </div>
                <span className="text-sm text-slate-300">Detection</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* CTA Section */}
        <div className="mt-20 text-center">
          <h2 className="mb-4 text-3xl font-bold text-white">Ready to Secure Your Network?</h2>
          <p className="mb-8 text-lg text-slate-300">
            Experience quantum-powered anomaly detection in action
          </p>
          <Link href="/network-anomaly">
            <Button size="lg" className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
              <Activity className="mr-2 h-5 w-5" />
              Start Monitoring Now
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}