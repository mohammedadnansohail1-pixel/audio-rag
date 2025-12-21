#!/usr/bin/env python
"""CLI for Audio RAG system."""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_rag import AudioRAG


def cmd_ingest(args):
    """Ingest audio files."""
    rag = AudioRAG.from_config(env=args.env, config_dir=args.config_dir)
    
    if args.cpu:
        rag.config.asr.device = "cpu"
        rag.config.asr.compute_type = "float32"
        rag.config.embedding.device = "cpu"
    
    for audio_path in args.files:
        print(f"\nIngesting: {audio_path}")
        try:
            result = rag.ingest(
                audio_path=audio_path,
                enable_diarization=not args.no_diarization,
                language=args.language,
            )
            print(f"  ✓ {result.num_chunks} chunks, {result.num_segments} segments")
            print(f"    Duration: {result.duration_seconds:.1f}s")
            print(f"    Speakers: {', '.join(result.speakers) or 'N/A'}")
            print(f"    Language: {result.language or 'unknown'}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    rag.unload_all()


def cmd_query(args):
    """Query the system."""
    rag = AudioRAG.from_config(env=args.env, config_dir=args.config_dir)
    
    if args.cpu:
        rag.config.embedding.device = "cpu"
    
    result = rag.query(
        query_text=args.query,
        top_k=args.top_k,
        generate_audio=args.audio,
        audio_output_path=args.output if args.audio else None,
    )
    
    print(f"\nQuery: {args.query}")
    print(f"Results: {len(result.results)}")
    print("-" * 50)
    
    for i, r in enumerate(result.results, 1):
        print(f"\n[{i}] Score: {r.score:.3f}")
        print(f"    Speaker: {r.chunk.speaker or 'Unknown'}")
        print(f"    Time: {r.chunk.start:.1f}s - {r.chunk.end:.1f}s")
        text = r.chunk.text[:200] + "..." if len(r.chunk.text) > 200 else r.chunk.text
        print(f"    Text: {text}")
    
    if result.audio_path:
        print(f"\n🔊 Audio saved: {result.audio_path}")
    
    rag.unload_all()


def cmd_status(args):
    """Show system status."""
    rag = AudioRAG.from_config(env=args.env, config_dir=args.config_dir)
    status = rag.status()
    
    print("\n=== Audio RAG Status ===\n")
    
    print("Configuration:")
    for key, value in status["config"].items():
        print(f"  {key}: {value}")
    
    print("\nResources:")
    res = status["resources"]
    print(f"  Max VRAM: {res['max_vram_gb']}GB")
    print(f"  Used VRAM: {res['used_vram_gb']}GB")
    print(f"  GPU Total: {res['gpu_info']['total_gb']}GB")
    print(f"  GPU Free: {res['gpu_info']['free_gb']}GB")
    
    print("\nCollection:")
    print(f"  Name: {status['collection']['name']}")
    print(f"  Vectors: {status['collection']['count']}")


def cmd_clear(args):
    """Clear the vector store."""
    if not args.yes:
        confirm = input("Are you sure you want to clear all data? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    
    rag = AudioRAG.from_config(env=args.env, config_dir=args.config_dir)
    rag.clear_collection()
    print("✓ Collection cleared")


def main():
    parser = argparse.ArgumentParser(
        description="Audio RAG - Config-driven Audio Retrieval System",
    )
    parser.add_argument("--env", "-e", default="development", help="Environment")
    parser.add_argument("--config-dir", "-c", default="configs", help="Config directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Ingest
    p = subparsers.add_parser("ingest", help="Ingest audio files")
    p.add_argument("files", nargs="+", help="Audio files")
    p.add_argument("--no-diarization", action="store_true", help="Skip diarization")
    p.add_argument("--language", "-l", help="Language code")
    p.set_defaults(func=cmd_ingest)
    
    # Query
    p = subparsers.add_parser("query", help="Query the system")
    p.add_argument("query", help="Query text")
    p.add_argument("--top-k", "-k", type=int, default=5, help="Results count")
    p.add_argument("--audio", "-a", action="store_true", help="Generate audio")
    p.add_argument("--output", "-o", default="./output/response.mp3", help="Audio path")
    p.set_defaults(func=cmd_query)
    
    # Status
    p = subparsers.add_parser("status", help="Show status")
    p.set_defaults(func=cmd_status)
    
    # Clear
    p = subparsers.add_parser("clear", help="Clear vector store")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    p.set_defaults(func=cmd_clear)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
