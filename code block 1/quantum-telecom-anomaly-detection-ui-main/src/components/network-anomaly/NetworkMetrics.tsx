"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts";

interface NetworkData {
  timestamp: number;
  latency: number;
  packetLoss: number;
  bandwidth: number;
  jitter: number;
  throughput: number;
}

interface NetworkMetricsProps {
  data: NetworkData[];
  isMonitoring: boolean;
}

export default function NetworkMetrics({ data, isMonitoring }: NetworkMetricsProps) {
  const chartData = data.map((d) => ({
    time: new Date(d.timestamp).toLocaleTimeString(),
    latency: d.latency.toFixed(2),
    packetLoss: d.packetLoss.toFixed(2),
    bandwidth: d.bandwidth.toFixed(2),
    throughput: d.throughput.toFixed(2),
  }));

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="text-white">Network Latency</CardTitle>
          <CardDescription className="text-slate-400">Real-time latency monitoring (ms)</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                  color: "#fff",
                }}
              />
              <Line type="monotone" dataKey="latency" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="text-white">Packet Loss Rate</CardTitle>
          <CardDescription className="text-slate-400">Network reliability indicator (%)</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                  color: "#fff",
                }}
              />
              <Line type="monotone" dataKey="packetLoss" stroke="#ef4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="text-white">Bandwidth Usage</CardTitle>
          <CardDescription className="text-slate-400">Network capacity utilization (Mbps)</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                  color: "#fff",
                }}
              />
              <Line type="monotone" dataKey="bandwidth" stroke="#10b981" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card className="border-slate-700/50 bg-slate-900/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="text-white">Throughput</CardTitle>
          <CardDescription className="text-slate-400">Data transfer rate (Mbps)</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                  color: "#fff",
                }}
              />
              <Line type="monotone" dataKey="throughput" stroke="#a855f7" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
