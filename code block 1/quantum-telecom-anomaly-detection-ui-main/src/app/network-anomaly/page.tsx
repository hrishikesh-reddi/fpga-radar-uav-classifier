"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Activity, AlertTriangle, Upload, Zap, Network, Brain, CheckCircle2, XCircle } from "lucide-react";
import NetworkMetrics from "@/components/network-anomaly/NetworkMetrics";
import QuantumCircuit from "@/components/network-anomaly/QuantumCircuit";
import AnomalyTimeline from "@/components/network-anomaly/AnomalyTimeline";
import FileUploader from "@/components/network-anomaly/FileUploader";
import ModelPredictions from "@/components/network-anomaly/ModelPredictions";

interface NetworkData {
  timestamp: number;
  latency: number;
  packetLoss: number;
  bandwidth: number;
  jitter: number;
  throughput: number;
}

interface PredictionResult {
  model: string;
  prediction: number;
  confidence: number;
  anomaly: boolean;
  processingTime: number;
}

export default function NetworkAnomalyPage() {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [networkData, setNetworkData] = useState<NetworkData[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [anomalyCount, setAnomalyCount] = useState(0);
  const [loading, setLoading] = useState(false);

  // Simulate real-time network data
  useEffect(() => {
    if (!isMonitoring) return;

    const interval = setInterval(() => {
      const newData: NetworkData = {
        timestamp: Date.now(),
        latency: Math.random() * 100 + 20,
        packetLoss: Math.random() * 5,
        bandwidth: Math.random() * 1000 + 500,
        jitter: Math.random() * 10,
        throughput: Math.random() * 800 + 200,
      };

      setNetworkData((prev) => [...prev.slice(-50), newData]);

      // Randomly trigger anomaly detection
      if (Math.random() > 0.7) {
        detectAnomaly(newData);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isMonitoring]);

  const detectAnomaly = async (data: NetworkData) => {
    setLoading(true);
    try {
      const response = await fetch("/api/anomaly-detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const result = await response.json();
        setPredictions((prev) => [...prev.slice(-10), ...result.predictions]);
        
        const hasAnomaly = result.predictions.some((p: PredictionResult) => p.anomaly);
        if (hasAnomaly) {
          setAnomalyCount((prev) => prev + 1);
        }
      }
    } catch (error) {
      console.error("Anomaly detection error:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/anomaly-detect/batch", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setPredictions(result.predictions);
        setAnomalyCount(result.predictions.filter((p: PredictionResult) => p.anomaly).length);
      }
    } catch (error) {
      console.error("Batch detection error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-white/10 bg-black/20 backdrop-blur-xl">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-purple-600">
                <Network className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Quantum Network Anomaly Detection</h1>
                <p className="text-sm text-slate-400">AI-powered telecom security monitoring</p>
              </div>
            </div>
            <Button
              onClick={() => setIsMonitoring(!isMonitoring)}
              className={isMonitoring ? "bg-red-600 hover:bg-red-700" : "bg-green-600 hover:bg-green-700"}
            >
              {isMonitoring ? (
                <>
                  <XCircle className="mr-2 h-4 w-4" />
                  Stop Monitoring
                </>
              ) : (
                <>
                  <CheckCircle2 className="mr-2 h-4 w-4" />
                  Start Monitoring
                </>
              )}
            </Button>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        {/* Status Cards */}
        <div className="mb-8 grid gap-4 md:grid-cols-4">
          <Card className="border-blue-500/20 bg-slate-900/50 backdrop-blur">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-slate-200">Monitoring Status</CardTitle>
              <Activity className={`h-4 w-4 ${isMonitoring ? "text-green-500" : "text-slate-500"}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">
                {isMonitoring ? "Active" : "Inactive"}
              </div>
              <p className="text-xs text-slate-400">
                {isMonitoring ? "Real-time analysis" : "Start to monitor"}
              </p>
            </CardContent>
          </Card>

          <Card className="border-purple-500/20 bg-slate-900/50 backdrop-blur">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-slate-200">Anomalies Detected</CardTitle>
              <AlertTriangle className="h-4 w-4 text-amber-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{anomalyCount}</div>
              <p className="text-xs text-slate-400">Total anomalies found</p>
            </CardContent>
          </Card>

          <Card className="border-green-500/20 bg-slate-900/50 backdrop-blur">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-slate-200">Data Points</CardTitle>
              <Zap className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{networkData.length}</div>
              <p className="text-xs text-slate-400">Network measurements</p>
            </CardContent>
          </Card>

          <Card className="border-amber-500/20 bg-slate-900/50 backdrop-blur">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-slate-200">Quantum Models</CardTitle>
              <Brain className="h-4 w-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">2</div>
              <p className="text-xs text-slate-400">QNN + QSVM active</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="dashboard" className="space-y-6">
          <TabsList className="bg-slate-900/50 backdrop-blur">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="quantum">Quantum Circuit</TabsTrigger>
            <TabsTrigger value="upload">Upload Data</TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard" className="space-y-6">
            {/* Network Metrics */}
            <NetworkMetrics data={networkData} isMonitoring={isMonitoring} />

            {/* Model Predictions & Timeline */}
            <div className="grid gap-6 lg:grid-cols-2">
              <ModelPredictions predictions={predictions} loading={loading} />
              <AnomalyTimeline predictions={predictions} />
            </div>
          </TabsContent>

          <TabsContent value="quantum" className="space-y-6">
            <QuantumCircuit />
          </TabsContent>

          <TabsContent value="upload" className="space-y-6">
            <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
              <CardHeader>
                <CardTitle className="text-white">Upload Network Data</CardTitle>
                <CardDescription className="text-slate-400">
                  Upload a CSV or JSON file with network traffic data for batch anomaly detection
                </CardDescription>
              </CardHeader>
              <CardContent>
                <FileUploader onUpload={handleFileUpload} loading={loading} />
              </CardContent>
            </Card>

            {predictions.length > 0 && (
              <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
                <CardHeader>
                  <CardTitle className="text-white">Batch Results</CardTitle>
                  <CardDescription className="text-slate-400">
                    Analysis results from uploaded data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {predictions.map((pred, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between rounded-lg border border-slate-700/50 bg-slate-800/30 p-3"
                      >
                        <div className="flex items-center gap-3">
                          <Badge variant={pred.anomaly ? "destructive" : "default"}>
                            {pred.model}
                          </Badge>
                          <span className="text-sm text-slate-300">
                            Confidence: {(pred.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <Badge variant={pred.anomaly ? "destructive" : "outline"}>
                          {pred.anomaly ? "Anomaly" : "Normal"}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
