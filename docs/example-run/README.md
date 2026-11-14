# Example run (evidence)

A real end-to-end run on 2026-11-14, with both MCP servers running and the free model `nvidia/nemotron-3.5-super-vl:free`:

- [run.log](run.log): the agent's terminal output (colours stripped). It shows discovery of both servers (20 + 4 tools), every MCP call, the model's proposals, the validation, the weather and the date probes.
- [plan.md](plan.md): the plan note the agent wrote to `demo-vault/Плани/` in Obsidian. A copy is kept here because the vault's agent output is git-ignored.

What happened:
1. Attempt 1, route `r8613349`, start 21 Nov: valid, but HIGH weather risk (snow above the tree line).
2. The code probed the other allowed start dates: 22 Nov high, 23 Nov medium.
3. Attempt 2 moved the start to 23 Nov: valid, medium risk, status `attention`.

Weather came from the recorded fixtures (`WEATHER_MODE=replay`).
