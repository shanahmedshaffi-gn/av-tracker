#!/usr/bin/env python3
"""
Batch audio file verification wrapper.

Processes all audio files in a directory recursively, performs speaker verification
on each, and generates a CSV summary report with statistics.

Usage:
  python scripts/run_verifier.py --embeddings data/embeddings --input data/demo --output results.csv --threshold 0.5
  python scripts/run_verifier.py -e data/embeddings -i data/demo -o results.csv

Output CSV columns:
  file_path, speaker_count, chunks_processed, identified_count, unknown_count,
  most_common_speaker, avg_similarity, top_similarity, bottom_similarity
"""

import argparse
import csv
import glob
import logging
from pathlib import Path
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multi_speaker_verification import MultiSpeakerVerifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_audio_files(
    embeddings_dir: str,
    input_dir: str,
    output_csv: str,
    threshold: float = 0.5,
    recursive: bool = True
) -> None:
    """
    Process all audio files in input directory and save summary to CSV.
    
    Args:
        embeddings_dir: Path to directory with .pkl speaker profiles
        input_dir: Path to directory containing audio files to process
        output_csv: Path to output CSV file
        threshold: Similarity threshold for verification (0.0-1.0)
        recursive: Whether to search recursively in subdirectories
    """
    
    # Validate inputs
    emb_path = Path(embeddings_dir)
    if not emb_path.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        sys.exit(1)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Initialize verifier
    logger.info(f"Initializing verifier with embeddings from {embeddings_dir}")
    try:
        verifier = MultiSpeakerVerifier(
            embeddings_dir,
            threshold=threshold
        )
    except Exception as e:
        logger.error(f"Failed to initialize verifier: {e}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(verifier.speakers)} speaker profiles")
    
    # Find all audio files
    pattern = "**/*.wav" if recursive else "*.wav"
    audio_files = sorted(input_path.glob(pattern))
    
    if not audio_files:
        logger.warning(f"No .wav files found in {input_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Process each file and collect results
    results = []
    
    for idx, audio_file in enumerate(audio_files, 1):
        logger.info(f"\n[{idx}/{len(audio_files)}] Processing: {audio_file.relative_to(input_path)}")
        
        try:
            # Process audio
            verifications = verifier.process_audio_file(str(audio_file))
            
            if not verifications:
                logger.warning(f"No results from {audio_file}")
                results.append({
                    'file_path': str(audio_file.relative_to(input_path)),
                    'speaker_count': len(verifier.speakers),
                    'chunks_processed': 0,
                    'identified_count': 0,
                    'unknown_count': 0,
                    'most_common_speaker': 'N/A',
                    'avg_similarity': 0.0,
                    'top_similarity': 0.0,
                    'bottom_similarity': 0.0
                })
                continue
            
            # Analyze results
            speaker_counts = defaultdict(int)
            similarities = []
            unknown_count = 0
            
            for _, speaker, similarity in verifications:
                similarities.append(similarity)
                if speaker == "Unknown":
                    unknown_count += 1
                else:
                    speaker_counts[speaker] += 1
            
            # Calculate statistics
            most_common = max(speaker_counts, default="N/A")
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            top_sim = max(similarities) if similarities else 0.0
            bottom_sim = min(similarities) if similarities else 0.0
            identified_count = len(similarities) - unknown_count
            
            result = {
                'file_path': str(audio_file.relative_to(input_path)),
                'speaker_count': len(verifier.speakers),
                'chunks_processed': len(verifications),
                'identified_count': identified_count,
                'unknown_count': unknown_count,
                'most_common_speaker': str(most_common),
                'avg_similarity': f"{avg_sim:.4f}",
                'top_similarity': f"{top_sim:.4f}",
                'bottom_similarity': f"{bottom_sim:.4f}"
            }
            
            results.append(result)
            
            logger.info(
                f"  ✓ Processed {len(verifications)} chunks: "
                f"{identified_count} identified, {unknown_count} unknown, "
                f"avg_sim={avg_sim:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            results.append({
                'file_path': str(audio_file.relative_to(input_path)),
                'speaker_count': len(verifier.speakers),
                'chunks_processed': 0,
                'identified_count': 0,
                'unknown_count': 0,
                'most_common_speaker': 'ERROR',
                'avg_similarity': 0.0,
                'top_similarity': 0.0,
                'bottom_similarity': 0.0
            })
    
    # Write CSV report
    if results:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nWriting summary report to: {output_path}")
        
        fieldnames = [
            'file_path',
            'speaker_count',
            'chunks_processed',
            'identified_count',
            'unknown_count',
            'most_common_speaker',
            'avg_similarity',
            'top_similarity',
            'bottom_similarity'
        ]
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            logger.info(f"✓ Report saved: {output_path}")
            logger.info(f"  Total files processed: {len(results)}")
            
            # Print summary table
            print("\n" + "=" * 100)
            print("SUMMARY REPORT")
            print("=" * 100)
            for row in results:
                print(f"File: {row['file_path']}")
                print(f"  Chunks: {row['chunks_processed']} | "
                      f"Identified: {row['identified_count']} | "
                      f"Unknown: {row['unknown_count']} | "
                      f"Avg Similarity: {row['avg_similarity']}")
            print("=" * 100 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to write CSV: {e}")
            sys.exit(1)
    else:
        logger.error("No results to write")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Batch verify audio files and generate summary report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process all .wav files in data/demo, save CSV
  python scripts/run_verifier.py -e data/embeddings -i data/demo -o results.csv
  
  # Custom threshold
  python scripts/run_verifier.py -e data/embeddings -i data/demo -o results.csv -t 0.6
  
  # Non-recursive search
  python scripts/run_verifier.py -e data/embeddings -i data/demo -o results.csv --no-recursive
        '''
    )
    
    parser.add_argument(
        '-e', '--embeddings',
        required=True,
        help='Path to embeddings directory with .pkl speaker profiles'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to directory containing audio files to process'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to output CSV file'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.5,
        help='Similarity threshold for verification (0.0-1.0, default: 0.5)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Search only in top-level directory (non-recursive)'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        parser.error(f"Threshold must be between 0.0 and 1.0, got {args.threshold}")
    
    logger.info(f"Starting batch verification")
    logger.info(f"  Embeddings: {args.embeddings}")
    logger.info(f"  Input: {args.input}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Threshold: {args.threshold}")
    logger.info(f"  Recursive: {not args.no_recursive}")
    
    process_audio_files(
        embeddings_dir=args.embeddings,
        input_dir=args.input,
        output_csv=args.output,
        threshold=args.threshold,
        recursive=not args.no_recursive
    )


if __name__ == '__main__':
    main()
