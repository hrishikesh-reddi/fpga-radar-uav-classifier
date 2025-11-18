"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, CheckCircle, Clock } from "lucide-react";

interface PredictionResult {
  model: string;
  prediction: number;
  confidence: number;
  anomaly: boolean;
  processingTime: number;
}

interface AnomalyTimelineProps {
  predictions: PredictionResult[];
}

export default function AnomalyTimeline({ predictions }: AnomalyTimelineProps) {
  const anomalies = predictions.filter((p) => p.anomaly).slice(-5);

  return (
    <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-white">
          <Clock className="h-5 w-5 text-amber-500" />
          Anomaly Timeline
        </CardTitle>
        <CardDescription className="text-slate-400">
          Recent anomalies detected in network traffic
        </CardDescription>
      </CardHeader>
      <CardContent>
        {anomalies.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <CheckCircle className="mb-3 h-12 w-12 text-green-500" />
            <p className="text-slate-400">No anomalies detected</p>
            <p className="text-sm text-slate-500">Network is operating normally</p>
          </div>
        ) : (
          <div className="space-y-3">
            {anomalies.map((anomaly, idx) => (
              <div
                key={idx}
                className="flex items-start gap-3 rounded-lg border border-red-500/20 bg-red-950/20 p-3"
              >
                <div className="mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-red-500/20">
                  <AlertTriangle className="h-4 w-4 text-red-400" />
                </div>
                <div className="flex-1">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="font-medium text-white">Network Anomaly Detected</span>
                    <Badge variant="destructive" className="text-xs">
                      {new Date().toLocaleTimeString()}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-slate-400">
                    <span>Model: {anomaly.model}</span>
                    <span>•</span>
                    <span>Confidence: {(anomaly.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
