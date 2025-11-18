"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Cpu, Zap, GitBranch } from "lucide-react";

export default function QuantumCircuit() {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-white">
            <Cpu className="h-5 w-5 text-blue-500" />
            Hybrid Quantum Neural Network
          </CardTitle>
          <CardDescription className="text-slate-400">
            Variational quantum circuit for pattern recognition
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Circuit Visualization */}
            <div className="rounded-lg border border-blue-500/30 bg-slate-950/50 p-6">
              <div className="space-y-4">
                {/* Qubit Lines */}
                {[0, 1, 2, 3].map((qubit) => (
                  <div key={qubit} className="flex items-center gap-2">
                    <Badge variant="outline" className="w-12 border-blue-500/50 text-blue-400">
                      q{qubit}
                    </Badge>
                    <div className="flex flex-1 items-center">
                      <div className="h-px flex-1 bg-blue-500/30" />
                      <div className="mx-2 flex h-8 w-8 items-center justify-center rounded border border-blue-500 bg-blue-500/20 text-xs text-blue-300">
                        H
                      </div>
                      <div className="h-px flex-1 bg-blue-500/30" />
                      <div className="mx-2 flex h-8 w-8 items-center justify-center rounded border border-purple-500 bg-purple-500/20 text-xs text-purple-300">
                        RY
                      </div>
                      <div className="h-px flex-1 bg-blue-500/30" />
                      {qubit < 3 && (
                        <>
                          <div className="mx-2 flex h-8 w-8 items-center justify-center rounded-full border border-green-500 bg-green-500/20">
                            <div className="h-2 w-2 rounded-full bg-green-400" />
                          </div>
                          <div className="h-px flex-1 bg-blue-500/30" />
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 text-sm">
              <div className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-3">
                <div className="text-slate-400">Qubits</div>
                <div className="text-lg font-bold text-white">4</div>
              </div>
              <div className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-3">
                <div className="text-slate-400">Layers</div>
                <div className="text-lg font-bold text-white">3</div>
              </div>
              <div className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-3">
                <div className="text-slate-400">Parameters</div>
                <div className="text-lg font-bold text-white">12</div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">Training Accuracy</span>
                <span className="font-semibold text-green-400">94.7%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                <div className="h-full w-[94.7%] bg-gradient-to-r from-green-500 to-emerald-400" />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-white">
            <GitBranch className="h-5 w-5 text-purple-500" />
            Quantum Support Vector Machine
          </CardTitle>
          <CardDescription className="text-slate-400">
            Quantum kernel method for classification
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Kernel Matrix Visualization */}
            <div className="rounded-lg border border-purple-500/30 bg-slate-950/50 p-6">
              <div className="grid grid-cols-8 gap-1">
                {Array.from({ length: 64 }).map((_, i) => {
                  const intensity = Math.random();
                  return (
                    <div
                      key={i}
                      className="aspect-square rounded-sm"
                      style={{
                        backgroundColor: `rgba(168, 85, 247, ${intensity})`,
                      }}
                    />
                  );
                })}
              </div>
              <p className="mt-3 text-center text-xs text-slate-400">Quantum Kernel Matrix</p>
            </div>

            <div className="grid grid-cols-3 gap-3 text-sm">
              <div className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-3">
                <div className="text-slate-400">Features</div>
                <div className="text-lg font-bold text-white">5</div>
              </div>
              <div className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-3">
                <div className="text-slate-400">Support Vectors</div>
                <div className="text-lg font-bold text-white">47</div>
              </div>
              <div className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-3">
                <div className="text-slate-400">Iterations</div>
                <div className="text-lg font-bold text-white">150</div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">Classification Accuracy</span>
                <span className="font-semibold text-purple-400">91.3%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                <div className="h-full w-[91.3%] bg-gradient-to-r from-purple-500 to-fuchsia-400" />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur lg:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-white">
            <Zap className="h-5 w-5 text-amber-500" />
            Model Architecture
          </CardTitle>
          <CardDescription className="text-slate-400">
            Hybrid classical-quantum pipeline for anomaly detection
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between gap-4">
            {/* Data Input */}
            <div className="flex-1 rounded-lg border border-slate-700/50 bg-slate-800/30 p-4 text-center">
              <div className="mb-2 text-2xl">📊</div>
              <div className="text-sm font-medium text-white">Network Data</div>
              <div className="text-xs text-slate-400">5 Features</div>
            </div>

            <div className="text-slate-600">→</div>

            {/* Classical Preprocessing */}
            <div className="flex-1 rounded-lg border border-blue-500/30 bg-blue-950/30 p-4 text-center">
              <div className="mb-2 text-2xl">⚙️</div>
              <div className="text-sm font-medium text-white">Preprocessing</div>
              <div className="text-xs text-slate-400">Normalization</div>
            </div>

            <div className="text-slate-600">→</div>

            {/* Quantum Layer */}
            <div className="flex-1 rounded-lg border border-purple-500/30 bg-purple-950/30 p-4 text-center">
              <div className="mb-2 text-2xl">⚛️</div>
              <div className="text-sm font-medium text-white">Quantum Layer</div>
              <div className="text-xs text-slate-400">QNN + QSVM</div>
            </div>

            <div className="text-slate-600">→</div>

            {/* Classical Output */}
            <div className="flex-1 rounded-lg border border-green-500/30 bg-green-950/30 p-4 text-center">
              <div className="mb-2 text-2xl">🎯</div>
              <div className="text-sm font-medium text-white">Classification</div>
              <div className="text-xs text-slate-400">Normal/Anomaly</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
