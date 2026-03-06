# treeroute-AIMS

An experiment applying Cook-Mertz space-efficient tree evaluation to shortest-path computation on road graphs, inspired by Ryan Williams' 2025 paper *"Simulating Time With Square-Root Space"* (arXiv:2502.17779).

## Motivation

Navigation software on low-cost edge devices common across East Africa operates under severe memory constraints. This experiment asks a simple question: if path cost computation is restructured as a tree evaluation problem, does the space-efficient algorithm from theoretical computer science translate into measurable memory savings in practice?

## What it does

The experiment represents a shortest path as a balanced binary tree of edge weights. It then evaluates that tree two ways: standard depth-first evaluation, which holds all child results in memory simultaneously, and Cook-Mertz evaluation, which processes one child at a time and holds only O(depth) values at once. Both are benchmarked against naive Dijkstra across road graphs of increasing size.

## Results

At 400 nodes, naive Dijkstra peaked at 96KB. The space-efficient tree evaluator held steady at 0.8KB across all graph sizes tested.

## Notes

This is an independent experiment, not a published result. A bug in the first implementation caused the space-efficient evaluator to silently discard accumulated values, returning only the last child's result. The bug was identified, corrected, and the experiment rerun. The corrected version is v2.

## Reference

Ryan Williams (2025). *Simulating Time With Square-Root Space*. arXiv:2502.17779. To appear at STOC 2025.
