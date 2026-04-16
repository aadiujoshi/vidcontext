import fs from 'fs';
import path from 'path';
import os from 'os';
import ffmpeg from 'fluent-ffmpeg';
// @ts-ignore
import ffmpegStatic from 'ffmpeg-static';
// @ts-ignore
import imghash from 'imghash';
import { GoogleGenAI } from '@google/genai';

if (ffmpegStatic) {
  ffmpeg.setFfmpegPath(ffmpegStatic);
}

export interface VidContextOptions {
  interval: number;
  apiKey: string;
  model?: string;
  prompt?: string;
  verbose?: boolean;
  hashThreshold?: number;
  retainFrames?: boolean;
}

export interface VidContextResult {
  timestamp: string | number;
  hash: string;
  description: string;
  imagePath?: string;
}

// Simple hamming distance function for hex strings
function hammingDistance(hash1: string, hash2: string): number {
  let distance = 0;
  for (let i = 0; i < hash1.length; i++) {
    const val1 = parseInt(hash1[i]!, 16);
    const val2 = parseInt(hash2[i]!, 16);
    let xor = val1 ^ val2;
    while (xor > 0) {
      distance += xor & 1;
      xor >>= 1;
    }
  }
  return distance;
}

const DEFAULT_PROMPT = "Describe this frame in extreme detail.";

export async function processVideo(
  videoPath: string,
  options: VidContextOptions
): Promise<VidContextResult[]> {
  const {
    interval,
    apiKey,
    model = 'gemini-1.5-flash',
    prompt = DEFAULT_PROMPT,
    verbose = false,
    hashThreshold = 10,
    retainFrames = false
  } = options;

  const log = (...args: any[]) => { if (verbose) console.log(...args); };

  if (!fs.existsSync(videoPath)) {
    throw new Error(`Video file not found at path: ${videoPath}`);
  }

  const ai = new GoogleGenAI({ apiKey });

  // Create temporary directory
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'vidcontext-'));
  log(`Created temporary directory for frames: ${tempDir}`);

  // Extract frames
  log(`Extracting frames with interval: ${interval}s...`);
  await new Promise<void>((resolve, reject) => {
    ffmpeg(videoPath)
      .outputOptions([`-vf fps=1/${interval}`])
      .output(path.join(tempDir, 'frame_%04d.jpg'))
      .on('end', () => {
        log('Frame extraction complete.');
        resolve();
      })
      .on('error', (err) => {
        reject(new Error(`Failed to extract frames: ${err.message}`));
      })
      .run();
  });

  const files = fs.readdirSync(tempDir)
    .filter(f => f.endsWith('.jpg'))
    .sort(); // Sorting ensures chronological order

  log(`Extracted ${files.length} frames.`);

  const results: VidContextResult[] = [];
  let lastHash: string | null = null;

  for (let i = 0; i < files.length; i++) {
    const file = files[i]!;
    const imagePath = path.join(tempDir, file);
    
    // Calculate timestamp based on index and interval
    // Note: this is an approximation depending on ffmpeg, 'fps=1/interval' means first frame at 0 or early.
    const timestampSecs = i * interval;

    log(`Processing frame ${i+1}/${files.length} (Timestamp: ~${timestampSecs}s)`);

    // Generate perceptual hash
    const hash = await imghash.hash(imagePath, 8, 'hex');
    
    // Check duplication
    if (lastHash) {
      const distance = hammingDistance(lastHash, hash);
      log(`Hamming distance to previous valid frame: ${distance}`);
      if (distance < hashThreshold) {
        log(`Skipping duplicate frame (distance ${distance} < threshold ${hashThreshold}).`);
        continue;
      }
    }

    lastHash = hash;

    // Send to Gemini
    log(`Frame is unique. Sending to Gemini...`);
    const imageBytes = fs.readFileSync(imagePath);
    const mimeType = 'image/jpeg';
    
    try {
      const response = await ai.models.generateContent({
        model,
        contents: [
          {
            role: 'user',
            parts: [
               { inlineData: { data: Buffer.from(imageBytes).toString('base64'), mimeType } },
               { text: prompt }
            ]
          }
        ]
      });

      const description = response.text || '';
      
      results.push({
        timestamp: timestampSecs,
        hash,
        description,
        ...(retainFrames ? { imagePath } : {})
      });
      log(`Received description (length ${description.length}).`);
      
    } catch (err: any) {
      log(`Error calling Gemini API for frame ${file}: ${err.message}`);
      // Throw to halt or could continue, but halting is safer for API billing / errors
      throw err;
    }
  }

  // Cleanup
  if (!retainFrames) {
    log(`Cleaning up temporary frames...`);
    fs.rmSync(tempDir, { recursive: true, force: true });
  } else {
    log(`Frames retained at ${tempDir}`);
  }

  log(`Video processing complete. Analyzed ${results.length} valid frames.`);
  return results;
}