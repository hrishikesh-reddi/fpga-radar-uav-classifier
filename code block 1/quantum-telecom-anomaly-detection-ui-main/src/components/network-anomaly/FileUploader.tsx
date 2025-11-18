"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Upload, FileText, X, Loader2 } from "lucide-react";

interface FileUploaderProps {
  onUpload: (file: File) => void;
  loading: boolean;
}

export default function FileUploader({ onUpload, loading }: FileUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  return (
    <div className="space-y-4">
      <div
        className={`relative rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
          dragActive
            ? "border-blue-500 bg-blue-950/20"
            : "border-slate-700 bg-slate-800/30 hover:border-slate-600"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept=".csv,.json,.npz"
          onChange={handleChange}
        />
        <label htmlFor="file-upload" className="cursor-pointer">
          <Upload className="mx-auto mb-4 h-12 w-12 text-slate-400" />
          <p className="mb-2 text-lg font-medium text-white">
            Drop your file here or click to browse
          </p>
          <p className="text-sm text-slate-400">
            Supports CSV, JSON, or NPZ files with network traffic data
          </p>
        </label>
      </div>

      {selectedFile && (
        <div className="flex items-center justify-between rounded-lg border border-slate-700 bg-slate-800/50 p-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-500/20">
              <FileText className="h-5 w-5 text-blue-400" />
            </div>
            <div>
              <p className="font-medium text-white">{selectedFile.name}</p>
              <p className="text-sm text-slate-400">
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSelectedFile(null)}
            disabled={loading}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      )}

      <Button
        onClick={handleUpload}
        disabled={!selectedFile || loading}
        className="w-full bg-blue-600 hover:bg-blue-700"
      >
        {loading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <Upload className="mr-2 h-4 w-4" />
            Analyze File
          </>
        )}
      </Button>
    </div>
  );
}
