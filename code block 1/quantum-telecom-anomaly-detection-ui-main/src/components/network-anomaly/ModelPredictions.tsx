"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, Loader2 } from "lucide-react";

interface PredictionResult {
  model: string;
  prediction: number;
  confidence: number;
  anomaly: boolean;
  processingTime: number;
}

interface ModelPredictionsProps {
  predictions: PredictionResult[];
  loading: boolean;
}

export default function ModelPredictions({ predictions, loading }: ModelPredictionsProps) {
  const latestPredictions = predictions.slice(-2);

  return (
    <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-white">
          <Brain className="h-5 w-5 text-purple-500" />
          Model Predictions
        </CardTitle>
        <CardDescription className="text-slate-400">
          Real-time quantum ML predictions
        </CardDescription>
      </CardHeader>
      <CardContent>
        {loading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
          </div>
        )}

        {!loading && predictions.length === 0 && (
          <div className="py-8 text-center text-slate-400">
            No predictions yet. Start monitoring to see results.
          </div>
        )}

        {!loading && latestPredictions.length > 0 && (
          <div className="space-y-4">
            {latestPredictions.map((pred, idx) => (
              <div
                key={idx}
                className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-4"
              >
                <div className="mb-3 flex items-center justify-between">
                  <Badge
                    variant="outline"
                    className={
                      pred.model === "Hybrid QNN"
                        ? "border-blue-500/50 text-blue-400"
                        : "border-purple-500/50 text-purple-400"
                    }
                  >
                    {pred.model}
                  </Badge>
                  <Badge
                    variant={pred.anomaly ? "destructive" : "default"}
                    className={pred.anomaly ? "" : "bg-green-600"}
                  >
                    {pred.anomaly ? "⚠️ Anomaly" : "✓ Normal"}
                  </Badge>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-400">Confidence</span>
                    <span className="font-semibold text-white">
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-slate-700">
                    <div
                      className={`h-full ${
                        pred.anomaly
                          ? "bg-gradient-to-r from-red-500 to-orange-400"
                          : "bg-gradient-to-r from-green-500 to-emerald-400"
                      }`}
                      style={{ width: `${pred.confidence * 100}%` }}
                    />
                  </div>

                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Processing Time</span>
                    <span>{pred.processingTime.toFixed(2)}ms</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {predictions.length > 2 && (
          <div className="mt-4 text-center text-xs text-slate-500">
            Showing latest 2 predictions ({predictions.length} total)
          </div>
        )}
      </CardContent>
    </Card>
  );
}
