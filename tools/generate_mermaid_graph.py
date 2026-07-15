#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dzeta-AGI Concept Link Visualizer: Generate Mermaid graphs from saved models."
    )
    parser.add_argument(
        "--model", required=True, help="Path to the saved .dzeta.bin model file"
    )
    parser.add_argument(
        "--token", required=True, help="Target token to inspect"
    )
    parser.add_argument(
        "--top", type=int, default=8, help="Number of semantic links to extract (default: 8)"
    )
    parser.add_argument(
        "--output", help="Optional markdown output file to write the Mermaid graph to"
    )
    parser.add_argument(
        "--inspector", default="./dzeta_inspect_model", help="Path to dzeta_inspect_model binary"
    )
    return parser.parse_args()

def run_inspector(inspector_path, model_path, token, top):
    # Verify inspector exists
    # If on Windows, check for .exe suffix
    if os.name == 'nt' and not inspector_path.endswith('.exe'):
        if os.path.exists(inspector_path + '.exe'):
            inspector_path += '.exe'
    
    if not os.path.exists(inspector_path):
        # Check standard compiled locations
        build_path = os.path.join("build", "dzeta_inspect_model")
        if os.name == 'nt':
            build_path += '.exe'
        if os.path.exists(build_path):
            inspector_path = build_path
        else:
            sys.exit(f"Error: Inspector binary not found at '{inspector_path}'. Please compile it first.")

    cmd = [inspector_path, "--model", model_path, "--token", token, "--top", str(top)]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(f"Error: Inspector failed with code {result.returncode}")
        
    return result.stdout

def parse_links(stdout):
    links = []
    in_links_block = False
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("token_links_begin"):
            in_links_block = True
            continue
        if line.startswith("token_links_end"):
            in_links_block = False
            continue
        if in_links_block:
            if line.startswith("rank") or not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    rank = parts[0]
                    target_token = parts[1]
                    score = float(parts[2])
                    links.append((target_token, score))
                except ValueError:
                    continue
    return links

def build_mermaid(source_token, links):
    if not links:
        return "%% No semantic links found for token: " + source_token
        
    lines = ["```mermaid", "graph TD"]
    # Define source node
    source_clean = source_token.replace('##', 'sub_')
    lines.append(f'    {source_clean}["{source_token}"]')
    
    for target, score in links:
        target_clean = target.replace('##', 'sub_')
        lines.append(f'    {target_clean}["{target}"]')
        lines.append(f'    {source_clean} -->|{score:.4f}| {target_clean}')
        
    lines.append("```")
    return "\n".join(lines)

def main():
    args = parse_args()
    stdout = run_inspector(args.inspector, args.model, args.token, args.top)
    links = parse_links(stdout)
    
    mermaid_graph = build_mermaid(args.token, links)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"# Semantic Link Graph for '{args.token}'\n\n")
            f.write(mermaid_graph + "\n")
        print(f"Graph written to '{args.output}'")
    else:
        print(mermaid_graph)

if __name__ == "__main__":
    main()
